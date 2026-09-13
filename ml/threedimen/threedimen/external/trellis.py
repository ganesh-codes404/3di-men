"""Imports a small, curated subset of (caption, mesh) pairs from
TRELLIS-500K (https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K) --
Microsoft's curated training corpus for their TRELLIS 3D generation model.

Scope, deliberately limited: TRELLIS-500K aggregates from six sources
(Objaverse-XL-Sketchfab, Objaverse-XL-GitHub, ABO, 3D-FUTURE, HSSD, Toys4k).
Only the two Objaverse-XL-sourced ones (~480K of ~500K entries) are
supported here, because they're the only ones retrievable through a
verified download path. Getting this API right mattered enough to verify
with a dedicated research pass before writing any of this code:

  - "objaverse-xl" is NOT a separate pip package to install -- that name
    exists on PyPI but is a stale, superseded release. The real,
    maintained path is the `objaverse.xl` submodule inside the same
    `objaverse` package used by cap3d.py: `import objaverse.xl as oxl`.
  - `oxl.get_annotations()` returns a DataFrame with columns including
    fileIdentifier, license, source, fileType, sha256 -- this is the
    authoritative source of "how do I actually download this object",
    which TRELLIS-500K's own CSVs don't provide (they give sha256 +
    file_identifier + captions, but not a download mechanism).
  - `oxl.download_objects(objects_df)` takes a DataFrame with
    fileIdentifier/source/sha256 columns and returns {fileIdentifier: path}.
    No API token needed for either GitHub or Sketchfab-sourced objects
    (Sketchfab-sourced meshes in this dataset are actually rehosted mirrors
    on Hugging Face, not fetched from sketchfab.com itself).

ABO/3D-FUTURE/HSSD/Toys4k-sourced entries use entirely different
hosting/download mechanisms and are NOT implemented here -- rows from those
sources are simply skipped, not silently mishandled.

The exact CSV filenames in the TRELLIS-500K repo weren't confirmed
character-for-character during research (only the naming *pattern*: one CSV
per source). Rather than hardcode a guessed filename that might be wrong,
this module lists the repo's actual files and matches by a normalized
substring, so a small filename spelling/casing difference doesn't silently
break this.
"""

from __future__ import annotations

from pathlib import Path

from .cap3d import ExternalSample, is_permissive

_TRELLIS_REPO = "JeffreyXiang/TRELLIS-500K"
_SUPPORTED_SOURCE_MARKERS = ("objaversexlsketchfab", "objaversexlgithub", "objaversexl")


def _normalize(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _find_supported_csv_files(cache_dir: str | Path | None = None) -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(_TRELLIS_REPO, repo_type="dataset")
    return [
        f for f in files
        if f.lower().endswith(".csv") and any(marker in _normalize(f) for marker in _SUPPORTED_SOURCE_MARKERS)
    ]


def _load_trellis_captions(cache_dir: str | Path | None = None):
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "TRELLIS import needs the optional extra: pip install -e '.[external]'"
        ) from exc

    csv_files = _find_supported_csv_files(cache_dir)
    if not csv_files:
        raise RuntimeError(
            f"couldn't find any Objaverse-XL-sourced CSV in {_TRELLIS_REPO} -- "
            "the repo's file layout may have changed since this was written; "
            "check https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K/tree/main"
        )

    frames = []
    for filename in csv_files:
        path = hf_hub_download(repo_id=_TRELLIS_REPO, filename=filename, repo_type="dataset", cache_dir=cache_dir)
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    return df[df["captions"].notna()] if "captions" in df.columns else df


def sample_trellis(
    n: int,
    *,
    seed: int = 0,
    check_license: bool = True,
    cache_dir: str | Path | None = None,
) -> list[ExternalSample]:
    """Pull n random (caption, mesh) pairs from TRELLIS-500K's
    Objaverse-XL-sourced entries. Requires the `objaverse`/`huggingface_hub`/
    `pandas` packages (pip install -e ".[external]")."""
    try:
        import objaverse.xl as oxl
    except ImportError as exc:
        raise ImportError(
            "TRELLIS import needs the optional extra: pip install -e '.[external]'"
        ) from exc
    import pandas as pd

    captions_df = _load_trellis_captions(cache_dir)
    if len(captions_df) == 0:
        return []
    shuffled = captions_df.sample(frac=1, random_state=seed).to_dict("records")

    annotations = oxl.get_annotations()  # fileIdentifier, license, source, fileType, sha256
    annotations_by_sha256 = {row["sha256"]: row for row in annotations.to_dict("records")}

    selected: list[dict] = []
    for row in shuffled:
        if len(selected) >= n:
            break
        ann = annotations_by_sha256.get(row.get("sha256"))
        if ann is None:
            continue  # captioned in TRELLIS-500K but not resolvable via objaverse.xl's current annotations
        if check_license:
            lic = str(ann.get("license") or "")
            if lic and not is_permissive(lic):
                continue
        selected.append({"row": row, "ann": ann})

    if not selected:
        return []

    objects_df = pd.DataFrame([s["ann"] for s in selected])
    downloaded = oxl.download_objects(objects_df)  # {fileIdentifier: local_path}

    samples = []
    for s in selected:
        file_id = s["ann"]["fileIdentifier"]
        mesh_path = downloaded.get(file_id)
        if mesh_path is None:
            continue
        samples.append(
            ExternalSample(
                uid=str(file_id),
                prompt=str(s["row"]["captions"]),
                mesh_path=Path(mesh_path),
                source="trellis-500k+objaverse-xl",
                metadata={"license": s["ann"].get("license", "unknown"), "sha256": s["row"].get("sha256")},
            )
        )
    return samples
