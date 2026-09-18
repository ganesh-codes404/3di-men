"""Stream Cap3D data and train on it locally, chunk by chunk, keeping
almost nothing on disk.

Why this exists (as opposed to `3dimen dataset import` + `model train`):
that path downloads meshes AND keeps a permanent DatasetStore copy of each
one (raw + processed + surface representation) -- measured at ~14.5MB/
sample on average, so a few thousand Cap3D samples means gigabytes on this
laptop's disk. This script never persists a full dataset at all: each
chunk of Cap3D objects is downloaded, converted to point clouds, trained on
immediately, and then dropped -- the raw meshes are deleted right after
processing (same as the Kaggle notebook's pipeline) and the processed
point clouds are never written anywhere, only held in memory for the
handful of epochs they're trained on before being discarded.

What DOES persist, and it's small and constant-size regardless of how much
of the corpus you work through:
  - the model checkpoint (a few MB), overwritten periodically so an
    interrupted run doesn't lose all training progress
  - a tiny list of already-processed UIDs (`local_run/cap3d_stream_state.jsonl`),
    so re-running this script later skips objects already trained on
    instead of starting over

Tuned for THIS laptop specifically: no CUDA, no MPS (Intel i5-5350U,
4 logical cores) -- a real local benchmark showed Chamfer distance at the
Kaggle notebook's default 2048 points/shape costs ~10s/batch on this CPU
(quadratic in point count), which would make even a modest Cap3D run take
days. Dropping to 256 points measured at ~0.65s/batch, a ~15x speedup,
which is what NUM_POINTS defaults to below.

Tradeoff worth knowing: because each chunk is trained on and then
discarded rather than kept in a persistent dataset the model can revisit
across many full epochs, this is closer to online/streaming learning than
the usual "shuffle the whole dataset every epoch" setup. It's a reasonable
trade for a hard storage constraint, but if quality matters more than
disk headroom, training epochs-over-a-persisted-dataset (the DatasetStore
path) would likely learn more from the same data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import trimesh
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `threedimen` imports without install

from threedimen.model.chamfer import chamfer_distance
from threedimen.model.device import resolve_device
from threedimen.model.networks import PointCloudDecoder, PointCloudEncoder, TextEncoder
from threedimen.model.vocab import Vocabulary

_CAP3D_REPO = "tiange/Cap3D"
_CAP3D_CAPTIONS_FILE = "Cap3D_automated_Objaverse_full.csv"
_PERMISSIVE_MARKERS = ("by", "cc0", "public-domain", "by-sa")

RUN_DIR = Path(__file__).resolve().parents[1] / "local_run"
CHECKPOINT_PATH = RUN_DIR / "threedimen_cap3d_stream.pt"
STATE_PATH = RUN_DIR / "cap3d_stream_state.jsonl"


def _is_permissive(lic: str) -> bool:
    return any(m in lic.lower() for m in _PERMISSIVE_MARKERS)


_TRANSIENT_NETWORK_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, OSError)


def _download_with_retry(objaverse_module, uids: list[str], max_retries: int = 5, backoff_seconds: float = 10.0) -> dict:
    """A single dropped connection (seen in practice: HTTP 499, "client closed
    request" -- a transient network hiccup, not a real failure) previously
    killed an entire multi-hour run outright, since objaverse.load_objects()
    had no retry logic of its own. Retries with linear backoff before
    finally giving up and letting the caller's try/finally save progress and
    exit cleanly (better than silently marking these UIDs as permanently
    done when we never actually found out whether they're downloadable)."""
    for attempt in range(1, max_retries + 1):
        try:
            with warnings.catch_warnings():
                # objaverse warns once per not-found uid -- real signal, but at
                # this dataset's miss rate (often >80%) it drowns out everything
                # else on screen. The actual found/skipped counts are tracked
                # and printed by the caller instead.
                warnings.filterwarnings("ignore", category=UserWarning, module="objaverse")
                return objaverse_module.load_objects(uids=uids, download_processes=1)
        except _TRANSIENT_NETWORK_ERRORS as e:
            if attempt == max_retries:
                raise
            wait = backoff_seconds * attempt
            tqdm.write(f"  download failed ({e!r}), retrying in {wait:.0f}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)


def _load_annotations_robust(objaverse_module, uids: list[str]) -> dict:
    """Same divide-and-conquer fallback as external/cap3d.py and the Kaggle
    notebook: objaverse.load_annotations() fails the WHOLE batch if even
    one UID is bad, so split and retry rather than one-by-one."""
    if not uids:
        return {}
    try:
        return objaverse_module.load_annotations(uids)
    except Exception:
        pass
    if len(uids) == 1:
        return {}
    mid = len(uids) // 2
    result = _load_annotations_robust(objaverse_module, uids[:mid])
    result.update(_load_annotations_robust(objaverse_module, uids[mid:]))
    return result


def _recenter_and_normalize(mesh: trimesh.Trimesh, target_diagonal: float = 1.0) -> trimesh.Trimesh:
    """MUST match geometry/ingest.py's convention -- mixing un-normalized
    points into the same batch as normalized ones previously blew up loss
    into the thousands/billions on a real run."""
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    if diagonal > 1e-9:
        mesh.apply_scale(target_diagonal / diagonal)
    return mesh


class _ChunkDataset(Dataset):
    def __init__(self, chunk_pairs: list[dict], vocab: Vocabulary, max_len: int):
        self.chunk_pairs = chunk_pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.chunk_pairs)

    def __getitem__(self, idx: int):
        pair = self.chunk_pairs[idx]
        tokens = torch.tensor(self.vocab.encode(pair["prompt"], self.max_len), dtype=torch.long)
        points = torch.tensor(pair["points"], dtype=torch.float32)
        return tokens, points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-points", type=int, default=256, help="points sampled per shape (benchmarked sweet spot on this CPU)")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--sub-batch-size", type=int, default=50, help="objects license-checked + downloaded together per network round (kept small for the license-check divide-and-conquer fallback -- see external/cap3d.py)")
    parser.add_argument("--max-chunk-mb", type=float, default=1024, help="a training chunk stops accumulating more objects once their downloaded size (measured for real, on disk, right before each is deleted) reaches this many MB -- default 1024 = 1GB")
    parser.add_argument("--max-chunk-objects", type=int, default=2000, help="safety cap on objects per chunk regardless of size, in case a chunk of unusually tiny objects would otherwise never hit --max-chunk-mb")
    parser.add_argument("--epochs-per-chunk", type=int, default=5, help="training epochs run over each chunk before it's discarded")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=None, help="stop after this many chunks this run (default: run until corpus exhausted or interrupted)")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="save the model every N chunks")
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(args.seed)
    print(f"device: {device} (this laptop has no CUDA/MPS, so this is CPU)")

    done_uids: set[str] = set()
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            for line in f:
                done_uids.add(json.loads(line)["uid"])
    print(f"{len(done_uids)} Cap3D objects already trained on in previous runs (skipping them)")

    print("downloading Cap3D captions CSV (text only, one-time, not per-run)...")
    captions_path = hf_hub_download(repo_id=_CAP3D_REPO, filename=_CAP3D_CAPTIONS_FILE, repo_type="dataset")
    captions_df = pd.read_csv(captions_path, header=None, names=["uid", "caption"])
    caption_by_uid = dict(zip(captions_df["uid"], captions_df["caption"]))
    total_corpus_size = len(captions_df)
    print(f"Cap3D corpus: {total_corpus_size} total captioned objects")

    # Model + vocab: if resuming from a checkpoint, reuse ITS vocab exactly
    # (the embedding table size must match) rather than rebuilding, which
    # could drift if the caption CSV changes between runs.
    if CHECKPOINT_PATH.exists():
        print(f"resuming model from {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        vocab = Vocabulary.from_dict(ckpt["vocab"])
        config = ckpt["config"]
    else:
        print("building vocabulary from the full Cap3D caption corpus (one-time, text-only)...")
        vocab = Vocabulary.build(captions_df["caption"].astype(str).tolist())
        config = {
            "num_points": args.num_points, "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim, "max_len": args.max_len, "lr": args.lr,
            "align_weight": args.align_weight, "seed": args.seed,
        }
    print(f"vocab size: {len(vocab)}")

    text_encoder = TextEncoder(len(vocab), config["embed_dim"], config["latent_dim"], pad_idx=vocab.pad_id).to(device)
    shape_encoder = PointCloudEncoder(config["hidden_dim"], config["latent_dim"]).to(device)
    shape_decoder = PointCloudDecoder(config["latent_dim"], config["num_points"], config["hidden_dim"]).to(device)
    if CHECKPOINT_PATH.exists():
        text_encoder.load_state_dict(ckpt["text_encoder"])
        shape_encoder.load_state_dict(ckpt["shape_encoder"])
        shape_decoder.load_state_dict(ckpt["shape_decoder"])

    params = list(text_encoder.parameters()) + list(shape_encoder.parameters()) + list(shape_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config["lr"])

    def save_checkpoint():
        torch.save(
            {
                "text_encoder": text_encoder.state_dict(),
                "shape_encoder": shape_encoder.state_dict(),
                "shape_decoder": shape_decoder.state_dict(),
                "vocab": vocab.to_dict(),
                "config": config,
            },
            CHECKPOINT_PATH,
        )

    rng = random.Random(args.seed)
    candidate_uids = captions_df["uid"].tolist()
    rng.shuffle(candidate_uids)
    candidate_uids = [u for u in candidate_uids if u not in done_uids]
    print(f"{len(candidate_uids)} not-yet-trained-on candidates remain")

    import objaverse  # imported late: only needed once we actually start downloading

    max_chunk_bytes = args.max_chunk_mb * 1024 * 1024

    rng_np = np.random.default_rng(args.seed)
    t0 = time.monotonic()
    chunk_num = 0
    trained_on_total = 0
    found_total = 0    # all-time, across this run: successfully downloaded + sampled
    skipped_total = 0  # all-time: license-skipped + not-found combined
    candidate_idx = 0
    state_f = open(STATE_PATH, "a")

    def cleanup_and_exit():
        state_f.close()
        save_checkpoint()
        print(f"\ncheckpoint saved to {CHECKPOINT_PATH}")
        print(f"resume-state saved to {STATE_PATH}")
        print(f"this run: {found_total} found, {skipped_total} skipped, {trained_on_total} trained-on")

    pbar = tqdm(total=len(candidate_uids), desc="Cap3D stream (download+train+discard)", unit="obj")
    try:
        while candidate_idx < len(candidate_uids):
            if args.max_chunks is not None and chunk_num >= args.max_chunks:
                tqdm.write(f"reached --max-chunks={args.max_chunks}, stopping")
                break

            # ---- build one chunk: pull sub-batches until the chunk's REAL
            # downloaded size (measured on disk, per-file, right before each
            # is deleted) reaches --max-chunk-mb, or the object-count safety
            # cap is hit, or candidates run out. Object sizes aren't known
            # until downloaded, so this can't be decided upfront -- it's
            # measured live as sub-batches come in. ----
            chunk_num += 1
            chunk_pairs: list[dict] = []
            chunk_uids_seen: list[str] = []
            chunk_bytes = 0
            license_skipped = 0
            not_found = 0

            while (
                candidate_idx < len(candidate_uids)
                and chunk_bytes < max_chunk_bytes
                and len(chunk_uids_seen) < args.max_chunk_objects
            ):
                sub_batch = candidate_uids[candidate_idx: candidate_idx + args.sub_batch_size]
                candidate_idx += len(sub_batch)
                chunk_uids_seen.extend(sub_batch)

                annotations = _load_annotations_robust(objaverse, sub_batch)
                permissive_sub_batch = []
                sub_license_skipped = 0
                for uid in sub_batch:
                    lic = str((annotations.get(uid) or {}).get("license") or "")
                    if lic and not _is_permissive(lic):
                        sub_license_skipped += 1
                        continue
                    permissive_sub_batch.append(uid)
                license_skipped += sub_license_skipped

                # objaverse writes to its own cache dir; every file below is
                # measured then deleted right after being read into points,
                # so peak disk usage is bounded by --max-chunk-mb, not by
                # how many total objects this run processes
                downloaded = _download_with_retry(objaverse, permissive_sub_batch)

                sub_found = 0
                sub_not_found = 0
                for uid in permissive_sub_batch:
                    mesh_path = downloaded.get(uid)
                    if mesh_path is None:
                        sub_not_found += 1
                        continue
                    try:
                        chunk_bytes += os.path.getsize(mesh_path)
                        mesh = trimesh.load(mesh_path, force="mesh")
                        mesh = _recenter_and_normalize(mesh)
                        points, _ = trimesh.sample.sample_surface(mesh, config["num_points"], seed=rng_np)
                        chunk_pairs.append({"uid": uid, "prompt": str(caption_by_uid[uid]), "points": np.asarray(points, dtype=np.float32)})
                        sub_found += 1
                    except Exception as e:
                        tqdm.write(f"  skipping {uid}: {e}")
                    finally:
                        # disk-safe: raw mesh gone the instant we're done reading it,
                        # whether or not it produced a usable sample
                        if mesh_path is not None and os.path.exists(mesh_path):
                            os.remove(mesh_path)
                not_found += sub_not_found

                found_total += sub_found
                skipped_total += sub_license_skipped + sub_not_found
                pbar.update(len(sub_batch))
                pbar.set_postfix(found=found_total, skipped=skipped_total, trained=trained_on_total)
                tqdm.write(
                    f"  sub-batch: {len(sub_batch)} attempted, {sub_found} found, "
                    f"{sub_license_skipped} license-skipped, {sub_not_found} not-found || "
                    f"ALL-TIME: {found_total} found, {skipped_total} skipped, {trained_on_total} trained"
                )

            # ---- train on this chunk, then discard it (nothing chunk-specific
            # was ever written to disk -- chunk_pairs only ever lived in RAM) ----
            chunk_recon = chunk_align = 0.0
            trained_on_this_chunk = len(chunk_pairs)
            if chunk_pairs:
                dataset = _ChunkDataset(chunk_pairs, vocab, config["max_len"])
                loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)
                for _ in range(args.epochs_per_chunk):
                    epoch_recon = epoch_align = 0.0
                    n_batches = 0
                    for tokens, points in loader:
                        tokens, points = tokens.to(device), points.to(device)
                        optimizer.zero_grad()
                        shape_latent = shape_encoder(points)
                        recon = shape_decoder(shape_latent)
                        recon_loss = chamfer_distance(recon, points)
                        text_latent = text_encoder(tokens)
                        align_loss = F.mse_loss(text_latent, shape_latent.detach())
                        (recon_loss + args.align_weight * align_loss).backward()
                        optimizer.step()
                        epoch_recon += recon_loss.item()
                        epoch_align += align_loss.item()
                        n_batches += 1
                    chunk_recon, chunk_align = epoch_recon / n_batches, epoch_align / n_batches
                trained_on_total += len(chunk_pairs)

            for uid in chunk_uids_seen:
                if uid not in done_uids:  # includes license-skipped/not-found/failed uids -- don't retry them forever either
                    done_uids.add(uid)
                    state_f.write(json.dumps({"uid": uid}) + "\n")
            state_f.flush()
            del chunk_pairs  # explicit: this run's chunk data is gone, not just out of scope

            if chunk_num % args.checkpoint_every == 0:
                save_checkpoint()

            elapsed_min = (time.monotonic() - t0) / 60
            pct_corpus = 100 * (len(done_uids)) / total_corpus_size
            chunk_mb = chunk_bytes / (1024 * 1024)
            # NOTE: the progress bar itself was already advanced sub-batch by
            # sub-batch above -- this only updates the displayed postfix numbers
            pbar.set_postfix(found=found_total, skipped=skipped_total, trained=trained_on_total, pct_corpus=f"{pct_corpus:.3f}%")
            tqdm.write(
                f"[chunk {chunk_num} complete] {len(chunk_uids_seen)} candidates ({chunk_mb:.1f}MB downloaded), "
                f"{license_skipped} license-skipped, {not_found} not-found, {trained_on_this_chunk} trained-on || "
                f"recon={chunk_recon:.4f} align={chunk_align:.4f} || "
                f"RUN TOTALS: {found_total} found, {skipped_total} skipped, {trained_on_total} trained || "
                f"ALL-TIME (incl. prior runs): {len(done_uids)} processed ({pct_corpus:.3f}% of {total_corpus_size}-object corpus) || "
                f"{elapsed_min:.1f}min elapsed"
            )
    except KeyboardInterrupt:
        tqdm.write("\ninterrupted -- saving progress before exiting")
    finally:
        pbar.close()
        cleanup_and_exit()


if __name__ == "__main__":
    main()
