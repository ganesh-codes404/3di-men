"""Imports a small, curated subset of (caption, mesh) pairs from Cap3D
(https://huggingface.co/datasets/tiange/Cap3D) -- large-scale automated
captions for Objaverse objects.

Deliberately pulls a small sample, not the full ~1.6M-object corpus (this
project's own principle: don't blindly download huge datasets -- see
docs/threedimen-architecture.md).

Two-step retrieval, because captions and meshes live in different places:
captions come from Cap3D's own CSV on Hugging Face; the meshes themselves
are downloaded separately via the `objaverse` package (Objaverse's own
hosting), keyed by the same object UID. Confirmed via
https://huggingface.co/datasets/tiange/Cap3D and
https://objaverse.allenai.org/docs/objaverse-1.0/ before writing this.

License note: Cap3D's captions are ODC-By 1.0. The underlying Objaverse 3D
assets carry MIXED, PER-OBJECT licenses -- some not approved for commercial
use. This module makes a best-effort attempt to read per-object license
metadata via objaverse.load_annotations() and skip non-permissive objects,
but that annotation schema was not independently verified against
objaverse's source before writing this (only load_objects() was confirmed)
-- treat check_license=True as a helpful filter, not a legal guarantee, and
verify license suitability yourself before using this data beyond local
experimentation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CAP3D_REPO = "tiange/Cap3D"
_CAP3D_CAPTIONS_FILE = "Cap3D_automated_Objaverse_full.csv"
_PERMISSIVE_LICENSE_MARKERS = ("by", "cc0", "public-domain", "by-sa")


@dataclass
class ExternalSample:
    uid: str
    prompt: str
    mesh_path: Path
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_captions_csv(cache_dir: str | Path | None = None):
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Cap3D import needs the optional extra: pip install -e '.[external]'"
        ) from exc

    path = hf_hub_download(
        repo_id=_CAP3D_REPO, filename=_CAP3D_CAPTIONS_FILE, repo_type="dataset", cache_dir=cache_dir
    )
    return pd.read_csv(path, header=None, names=["uid", "caption"])


def is_permissive(license_str: str) -> bool:
    lic = license_str.lower()
    return any(marker in lic for marker in _PERMISSIVE_LICENSE_MARKERS)


def _load_annotations_robust(objaverse_module, uids: list[str]) -> dict[str, Any]:
    """objaverse.load_annotations() raises for the whole batch if even one
    requested UID can't be resolved (confirmed by hitting this directly on
    real Cap3D data -- unlike load_objects(), which just warns and skips
    unresolvable UIDs). Try the fast batched call first; only fall back to
    resolving one UID at a time -- so a single bad UID doesn't silently
    disable license info for every other UID in the batch -- if the batch
    call actually fails."""
    try:
        return objaverse_module.load_annotations(uids)
    except Exception:
        pass
    result: dict[str, Any] = {}
    for uid in uids:
        try:
            result.update(objaverse_module.load_annotations([uid]))
        except Exception:
            continue  # this one UID genuinely can't be resolved -- skip it, not the whole batch
    return result


def sample_cap3d(
    n: int,
    *,
    seed: int = 0,
    check_license: bool = True,
    cache_dir: str | Path | None = None,
) -> list[ExternalSample]:
    """Pull n random (caption, mesh) pairs from Cap3D/Objaverse.

    Requires the `objaverse`/`huggingface_hub`/`pandas` packages
    (pip install -e ".[external]"). Meshes download to objaverse's own
    cache directory (~/.objaverse by default), not this project's dataset
    store -- callers run the results through ingest_mesh()/DatasetStore
    same as any other source (see cli/main.py's `dataset import` command).
    """
    try:
        import objaverse
    except ImportError as exc:
        raise ImportError(
            "Cap3D import needs the optional extra: pip install -e '.[external]'"
        ) from exc

    captions_df = _load_captions_csv(cache_dir)
    rng = random.Random(seed)
    candidate_uids = captions_df["uid"].tolist()
    rng.shuffle(candidate_uids)

    license_by_uid: dict[str, str] = {}
    if check_license:
        # Over-fetch a bit of headroom since some candidates may get
        # filtered out by license below.
        annotations = _load_annotations_robust(objaverse, candidate_uids[: max(n * 5, 50)])
        for uid, ann in annotations.items():
            license_by_uid[uid] = str((ann or {}).get("license") or "")

    caption_by_uid = dict(zip(captions_df["uid"], captions_df["caption"]))

    selected: list[str] = []
    for uid in candidate_uids:
        if len(selected) >= n:
            break
        if check_license and license_by_uid:
            lic = license_by_uid.get(uid, "")
            if lic and not is_permissive(lic):
                continue
        selected.append(uid)

    downloaded = objaverse.load_objects(uids=selected, download_processes=1)

    samples = []
    for uid in selected:
        mesh_path = downloaded.get(uid)
        if mesh_path is None:
            continue
        samples.append(
            ExternalSample(
                uid=uid,
                prompt=str(caption_by_uid[uid]),
                mesh_path=Path(mesh_path),
                source="cap3d+objaverse",
                metadata={"license": license_by_uid.get(uid, "unknown")},
            )
        )
    return samples
