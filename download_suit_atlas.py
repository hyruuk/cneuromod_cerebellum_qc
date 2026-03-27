#!/usr/bin/env python3
"""
Download the SUIT cerebellar lobular atlas (Diedrichsen 2009) in MNI space.

Usage:
    python download_suit_atlas.py --output_dir /path/to/atlases

The atlas NIfTI and label TSV are saved to output_dir.
After downloading, run cereb_qc.py with:
    --suit_atlas /path/to/atlases/suit_atl-Anatom_space-MNI_dseg.nii.gz
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve


# ---------------------------------------------------------------------------
# Atlas source URLs (Diedrichsen Lab, GitHub)
# ---------------------------------------------------------------------------

ATLAS_URLS = {
    # Primary: discrete label atlas (28 cerebellar lobules) in MNI space
    "atlas_nii": (
        "https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/"
        "Diedrichsen_2009/atl-Anatom_space-MNI_dseg.nii"
    ),
    # Label table (TSV: index, name, color)
    "atlas_tsv": (
        "https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/"
        "Diedrichsen_2009/atl-Anatom_space-MNI_dseg.tsv"
    ),
}

OUTPUT_NAMES = {
    "atlas_nii": "suit_atl-Anatom_space-MNI_dseg.nii",
    "atlas_tsv": "suit_atl-Anatom_space-MNI_dseg.tsv",
}

# Expected MNI template space dimensions at 1mm (SUIT is 1mm MNI152)
EXPECTED_SHAPE_APPROX = (91, 109, 91)  # MNI152 standard 1mm


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, 100 * downloaded // total_size)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        sys.stdout.write(f"\r  [{bar}] {pct}% ({downloaded // 1024} KB / {total_size // 1024} KB)")
        sys.stdout.flush()
    if downloaded >= total_size:
        print()


def download_file(url: str, dest: Path) -> None:
    print(f"  Downloading: {url}")
    print(f"  To: {dest}")
    urlretrieve(url, dest, reporthook=_progress_hook)


def verify_atlas(nii_path: Path) -> bool:
    """Load the atlas and verify it contains expected cerebellar labels."""
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(nii_path)
        data = np.asarray(img.dataobj, dtype=np.int16)
        present = set(np.unique(data)) - {0}
        expected = set(range(1, 29))  # labels 1–28
        missing = expected - present
        if missing:
            print(f"  WARNING: Missing labels after download: {sorted(missing)}")
            print("  The atlas may be in a different version with different label IDs.")
            print("  This is not necessarily an error — check the TSV label file.")
        else:
            print(f"  ✓ All 28 expected labels found.")
        print(f"  Atlas shape: {img.shape}, dtype: {img.get_data_dtype()}")
        print(f"  Voxel size: {img.header.get_zooms()}")
        return True
    except Exception as e:
        print(f"  ERROR verifying atlas: {e}")
        return False


def compress_nii(nii_path: Path) -> Path:
    """Compress a .nii file to .nii.gz and return the new path."""
    import gzip
    import shutil

    # Append .gz to the full filename: foo.nii → foo.nii.gz
    gz_path = nii_path.parent / (nii_path.name + ".gz")
    with open(nii_path, "rb") as f_in, gzip.open(str(gz_path), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    nii_path.unlink()
    return gz_path


def write_label_tsv(output_dir: Path) -> Path:
    """
    Write the SUIT label TSV from the built-in SUIT_LABEL_MAP.

    The Diedrichsen lab GitHub TSV file may not be publicly accessible;
    we generate it from the known label map instead.
    """
    from qc.atlas import SUIT_LABEL_MAP

    tsv_path = output_dir / OUTPUT_NAMES["atlas_tsv"]
    with open(tsv_path, "w") as f:
        f.write("index\tname\n")
        f.write("0\tBackground\n")
        for idx, name in sorted(SUIT_LABEL_MAP.items()):
            f.write(f"{idx}\t{name}\n")
    print(f"  Label TSV generated from built-in map: {tsv_path}")
    return tsv_path


def main():
    parser = argparse.ArgumentParser(
        description="Download SUIT cerebellar atlas (Diedrichsen 2009) in MNI space."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the atlas files will be saved.",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip atlas verification (nibabel not required).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SUIT Cerebellar Atlas Downloader ===")
    print(f"Output directory: {output_dir}\n")

    # Download NIfTI
    nii_dest = output_dir / OUTPUT_NAMES["atlas_nii"]
    # compress_nii appends .gz to the full filename: foo.nii → foo.nii.gz
    gz_dest = output_dir / (OUTPUT_NAMES["atlas_nii"] + ".gz")

    if gz_dest.exists():
        print(f"✓ Atlas already exists: {gz_dest}")
        print("  Delete the file and re-run to re-download.")
    elif nii_dest.exists():
        # Downloaded but not yet compressed (previous interrupted run)
        print("  Compressing existing .nii to .nii.gz...")
        gz_dest = compress_nii(nii_dest)
        print(f"  Saved: {gz_dest}")
    else:
        download_file(ATLAS_URLS["atlas_nii"], nii_dest)
        print("  Compressing to .nii.gz...")
        gz_dest = compress_nii(nii_dest)
        print(f"  Saved: {gz_dest}")

    # Generate label TSV from built-in map (GitHub TSV URL is unreliable)
    tsv_dest = output_dir / OUTPUT_NAMES["atlas_tsv"]
    if tsv_dest.exists():
        print(f"✓ Label TSV already exists: {tsv_dest}")
    else:
        write_label_tsv(output_dir)

    # Verify
    if not args.no_verify:
        print("\n  Verifying atlas integrity...")
        verify_atlas(gz_dest)

    print(f"\n✓ Done. Use with cereb_qc.py:")
    print(f"  --suit_atlas {gz_dest}\n")


if __name__ == "__main__":
    main()
