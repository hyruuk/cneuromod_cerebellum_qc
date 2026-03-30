#!/usr/bin/env python3
"""
CNeuroMod — Cerebellar QC Report Generator

Generates a single interactive HTML QC report for all subjects,
covering tSNR, motion, brain mask coverage, physiological noise,
and advanced metrics relevant to cerebellar analysis during gameplay.

Usage:
    # Full mode (requires BOLD files fetched via datalad):
    python cereb_qc.py \\
        --fmriprep_dir /path/to/{dataset}.fmriprep \\
        --output_dir /path/to/output \\
        --suit_atlas /path/to/suit_atlas.nii.gz

    # Fast confounds-only mode (no BOLD needed):
    python cereb_qc.py \\
        --fmriprep_dir /path/to/{dataset}.fmriprep \\
        --output_dir /path/to/output \\
        --suit_atlas /path/to/suit_atlas.nii.gz \\
        --no_bold

    # Dry run (check file availability only):
    python cereb_qc.py \\
        --fmriprep_dir /path/to/{dataset}.fmriprep \\
        --suit_atlas /path/to/suit_atlas.nii.gz \\
        --dry_run

    # Single subject, specific sessions:
    python cereb_qc.py \\
        --fmriprep_dir /path/to/{dataset}.fmriprep \\
        --output_dir /path/to/output \\
        --suit_atlas /path/to/suit_atlas.nii.gz \\
        --subjects sub-01 \\
        --n_jobs 2
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cerebellar QC report for an fmriprep dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fmriprep_dir",
        required=True,
        help="Root directory of the fmriprep dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory where the HTML report will be written. Required unless --dry_run.",
    )
    parser.add_argument(
        "--suit_atlas",
        default="data/suit_atl-Anatom_space-MNI_dseg.nii.gz",
        help=(
            "Path to the SUIT cerebellar lobular atlas NIfTI file (discrete integer labels, "
            "in MNI152 space). Run download_suit_atlas.py to obtain this file."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="SUB_ID",
        help="Restrict to specific subjects (e.g., sub-01 sub-02). Default: all.",
    )
    parser.add_argument(
        "--fd_threshold",
        type=float,
        default=0.5,
        help="Framewise displacement threshold for scrubbing (mm).",
    )
    parser.add_argument(
        "--dvars_threshold",
        type=float,
        default=1.5,
        help="Standardized DVARS threshold for scrubbing.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel workers (joblib). Each BOLD worker uses ~4 GB RAM. "
            "Recommend n_jobs=1 for local testing, 4-8 on HPC."
        ),
    )
    parser.add_argument(
        "--no_bold",
        action="store_true",
        help="Skip BOLD loading. Compute only confounds-derived metrics (fast mode).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print file availability table and exit without computing anything.",
    )
    parser.add_argument(
        "--report_name",
        default="cerebellar_qc_report.html",
        help="Name of the output HTML file.",
    )
    return parser.parse_args()


def check_ram_warning(n_jobs: int) -> None:
    """Warn if estimated RAM usage may be excessive."""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        estimated_gb = n_jobs * 4.5  # ~4.5 GB per worker (float32 BOLD + tSNR + atlas)
        if estimated_gb > available_gb * 0.8:
            warnings.warn(
                f"Estimated RAM for {n_jobs} workers: ~{estimated_gb:.0f} GB. "
                f"Available: {available_gb:.0f} GB. Consider reducing --n_jobs.",
                ResourceWarning,
                stacklevel=2,
            )
    except ImportError:
        pass  # psutil not available, skip check


def main() -> None:
    args = parse_args()

    # Validate inputs
    fmriprep_dir = Path(args.fmriprep_dir)
    if not fmriprep_dir.is_dir():
        sys.exit(f"ERROR: fmriprep_dir does not exist: {fmriprep_dir}")

    suit_atlas_path = Path(args.suit_atlas)
    atlas_required = not args.dry_run and not args.no_bold
    if atlas_required and not suit_atlas_path.exists():
        sys.exit(
            f"ERROR: SUIT atlas not found: {suit_atlas_path}\n"
            "Run: python download_suit_atlas.py --output_dir /your/atlas/dir\n"
            "Tip: use --no_bold to run in confounds-only mode without the atlas."
        )

    # output_dir has a default ("output"), so this guard is only for the edge case
    # where the user explicitly passes empty string
    if not args.dry_run and not args.output_dir:
        sys.exit("ERROR: --output_dir cannot be empty.")

    output_dir = Path(args.output_dir) if args.output_dir else None

    # -----------------------------------------------------------------------
    # Step 1: Discover runs
    # -----------------------------------------------------------------------
    print("\n[1/6] Discovering runs...")
    from qc.discovery import discover_runs, print_availability_report

    runs = discover_runs(fmriprep_dir, subjects=args.subjects)
    print(f"  Found {len(runs)} runs across {len(set(r.subject for r in runs))} subjects.")

    if args.dry_run:
        print_availability_report(runs)
        return

    # -----------------------------------------------------------------------
    # Step 2: Load SUIT atlas (once, shared across all workers)
    # -----------------------------------------------------------------------
    if args.no_bold:
        print("\n[2/6] Skipping SUIT atlas (--no_bold mode, confounds only).")
        suit_data = np.zeros((1, 1, 1), dtype=np.int16)
        suit_labels = {}
    else:
        print("\n[2/6] Loading SUIT atlas...")
        from qc.atlas import load_suit_atlas

        # Use first available BOLD for reference grid, or brain mask
        ref_img = None
        for r in runs:
            if r.mask_available:
                ref_img = nib.load(r.mask_path)
                print(f"  Reference grid from: {r.mask_path.name}")
                break
            if r.bold_available:
                # Use a 3D slice (avoid loading full 4D)
                img_4d = nib.load(r.bold_path)
                ref_img = nib.Nifti1Image(
                    np.zeros(img_4d.shape[:3]),
                    img_4d.affine,
                    img_4d.header,
                )
                print(f"  Reference grid from: {r.bold_path.name}")
                break

        if ref_img is None:
            warnings.warn(
                "No brain mask or BOLD files available locally. "
                "Using default 97×115×97 MNI grid for atlas resampling. "
                "Atlas ROI metrics will be approximate.",
                UserWarning,
            )
            ref_img = _make_default_mni_ref()

        suit_data, suit_labels = load_suit_atlas(
            suit_atlas_path,
            ref_img,
            cache_dir=output_dir,
        )
        print(f"  SUIT atlas loaded: shape={suit_data.shape}, {len(suit_labels)} lobules.")

    # -----------------------------------------------------------------------
    # Step 3: Pre-load aseg per subject (loaded once, shared across runs)
    # -----------------------------------------------------------------------
    print("\n[3/6] Loading per-subject aseg segmentations...")
    from qc.atlas import load_subject_aseg

    subject_aseg: Dict[str, tuple] = {}
    for subj in sorted(set(r.subject for r in runs)):
        # Find any functional directory for this subject
        subj_runs = [r for r in runs if r.subject == subj]
        loaded = False
        for r in subj_runs:
            aseg_data, aseg_img = load_subject_aseg(r.func_dir, r.subject, r.session)
            if aseg_data is not None:
                subject_aseg[subj] = (aseg_data, aseg_img)
                print(f"  {subj}: aseg loaded ({r.session})")
                loaded = True
                break
        if not loaded:
            subject_aseg[subj] = (None, None)
            print(f"  {subj}: aseg unavailable (datalad not fetched)")

    # -----------------------------------------------------------------------
    # Step 4: Identify representative runs (for carpet + inter-lobule corr)
    # -----------------------------------------------------------------------
    print("\n[4/6] Processing runs...")
    from qc.aggregation import identify_representative_runs, process_run, aggregate_results

    check_ram_warning(args.n_jobs)

    # We need at least the confounds to know which runs are worst/best motion.
    # For now, designate first available BOLD run per subject as representative.
    # After processing, we'll re-identify and re-compute carpet for worst/best.
    representative_runs: Dict[str, set] = {}  # {subject: set of (session, run) tuples}
    for subj in set(r.subject for r in runs):
        subj_runs = [r for r in runs if r.subject == subj and r.bold_available]
        if subj_runs:
            # Placeholder: use first; will update after processing
            representative_runs[subj] = {(subj_runs[0].session, subj_runs[0].run)}

    def _process(r: "RunInfo") -> dict:
        aseg_data, aseg_img = subject_aseg.get(r.subject, (None, None))
        is_rep = (r.session, r.run) in representative_runs.get(r.subject, set())
        return process_run(
            run_info=r,
            suit_data=suit_data,
            aseg_data=aseg_data,
            aseg_img=aseg_img,
            fd_threshold=args.fd_threshold,
            dvars_threshold=args.dvars_threshold,
            no_bold=args.no_bold,
            extract_carpet=is_rep and not args.no_bold,
            extract_interlobule=is_rep and not args.no_bold,
        )

    if args.n_jobs == 1:
        # Sequential (easier to debug, no pickle issues)
        run_results = []
        total = len(runs)
        for i, r in enumerate(runs, 1):
            print(f"  [{i:3d}/{total}] {r.subject} {r.session} {r.run}...", end=" ", flush=True)
            result = _process(r)
            run_results.append(result)
            cereb_tsnr = result.get("cereb_mean", float("nan"))
            mean_fd = result.get("mean_fd", float("nan"))
            status = []
            if np.isfinite(cereb_tsnr):
                status.append(f"tSNR={cereb_tsnr:.0f}")
            if np.isfinite(mean_fd):
                status.append(f"FD={mean_fd:.3f}")
            print(", ".join(status) if status else "no data")
    else:
        from joblib import Parallel, delayed
        run_results = Parallel(n_jobs=args.n_jobs, prefer="processes", verbose=5)(
            delayed(_process)(r) for r in runs
        )

    # After initial processing, update representative runs based on motion
    # and re-process those runs with carpet/interlobule flags
    print("\n  Identifying representative runs (worst/best motion per subject)...")
    rep_map = identify_representative_runs(run_results)

    # Re-process representative runs with carpet + interlobule enabled (if not already)
    for r in runs:
        if args.no_bold or not r.bold_available:
            continue
        subj = r.subject
        run_key = f"{r.session}/{r.run}"
        if subj not in rep_map:
            continue
        if run_key not in (rep_map[subj].get("best"), rep_map[subj].get("worst")):
            continue
        # Find and replace the result for this run
        for i, res in enumerate(run_results):
            if res["subject"] == subj and res["session"] == r.session and res["run"] == r.run:
                if res.get("_carpet") is None or res.get("_interlobule_corr") is None:
                    print(f"  Re-processing {subj} {r.session} {r.run} for carpet/interlobule...", flush=True)
                    aseg_data, aseg_img = subject_aseg.get(subj, (None, None))
                    run_results[i] = process_run(
                        run_info=r,
                        suit_data=suit_data,
                        aseg_data=aseg_data,
                        aseg_img=aseg_img,
                        fd_threshold=args.fd_threshold,
                        dvars_threshold=args.dvars_threshold,
                        no_bold=False,
                        extract_carpet=True,
                        extract_interlobule=True,
                    )
                break

    # -----------------------------------------------------------------------
    # Step 5: Aggregate results
    # -----------------------------------------------------------------------
    print("\n[5/6] Aggregating results...")
    dfs = aggregate_results(run_results)
    df_runs = dfs["runs"]
    df_sessions = dfs["sessions"]
    df_subjects = dfs["subjects"]

    n_with_data = int(df_runs["bold_available"].sum()) if "bold_available" in df_runs.columns else 0
    print(f"  Runs processed: {len(df_runs)} total, {n_with_data} with BOLD data.")

    if not df_runs.empty and "cereb_mean" in df_runs.columns:
        print("\n  Subject-level tSNR summary:")
        for subj in sorted(df_subjects["subject"]):
            row = df_subjects[df_subjects["subject"] == subj].iloc[0]
            tsnr = row.get("cereb_mean", float("nan"))
            fd = row.get("mean_fd", float("nan"))
            usable = row.get("pct_usable", float("nan"))
            tsnr_str = f"{tsnr:.1f}" if np.isfinite(tsnr) else "N/A"
            fd_str = f"{fd:.3f}" if np.isfinite(fd) else "N/A"
            usable_str = f"{usable:.1f}%" if np.isfinite(usable) else "N/A"
            print(f"    {subj}: cereb tSNR={tsnr_str}, mean FD={fd_str}, usable={usable_str}")

    # Average tSNR maps across all runs per subject
    print("\n  Averaging tSNR maps across runs per subject...")
    _tsnr_stacks: Dict[str, list] = {}
    _tsnr_ref_img: Dict[str, object] = {}
    for r in run_results:
        subj = r["subject"]
        img = r.get("_tsnr_img")
        if img is None:
            continue
        if subj not in _tsnr_stacks:
            _tsnr_stacks[subj] = []
            _tsnr_ref_img[subj] = img
        _tsnr_stacks[subj].append(img.get_fdata(dtype=np.float32))

    tsnr_imgs: Dict[str, object] = {}
    for subj, arrays in _tsnr_stacks.items():
        n = len(arrays)
        stack = np.stack(arrays, axis=-1)  # (x, y, z, n_runs)
        mean_tsnr = np.nanmean(stack, axis=-1).astype(np.float32)
        tsnr_imgs[subj] = nib.Nifti1Image(mean_tsnr, _tsnr_ref_img[subj].affine)
        print(f"    {subj}: averaged {n} tSNR maps")

    # Save aggregated DataFrames as CSV for further analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    dfs["runs"].to_csv(output_dir / "qc_runs.csv", index=False)
    dfs["sessions"].to_csv(output_dir / "qc_sessions.csv", index=False)
    dfs["subjects"].to_csv(output_dir / "qc_subjects.csv", index=False)
    print(f"\n  CSVs saved to: {output_dir}")

    # -----------------------------------------------------------------------
    # Step 6: Generate HTML report
    # -----------------------------------------------------------------------
    print("\n[6/6] Generating HTML report...")
    from qc.report.builder import generate_html_report

    report_path = output_dir / args.report_name
    generate_html_report(
        dfs=dfs,
        run_results=run_results,
        tsnr_imgs=tsnr_imgs,
        output_path=report_path,
    )

    print(f"\n✓ Done! Open in browser: {report_path}")


def _make_default_mni_ref() -> nib.Nifti1Image:
    """
    Create a minimal reference image for MNI152NLin2009cAsym 2mm space
    (97×115×97 voxels) to use when no real data is available for resampling.
    """
    # MNI152NLin2009cAsym 2mm affine (standard)
    affine = np.array([
        [-2., 0., 0., 90.],
        [0., 2., 0., -126.],
        [0., 0., 2., -72.],
        [0., 0., 0., 1.],
    ])
    data = np.zeros((97, 115, 97), dtype=np.uint8)
    return nib.Nifti1Image(data, affine)


if __name__ == "__main__":
    main()
