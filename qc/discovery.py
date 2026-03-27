"""
BIDS-like file discovery for mario.fmriprep dataset.

Discovers all subjects/sessions/runs and checks datalad/git-annex availability
by resolving symlinks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class RunInfo:
    subject: str
    session: str
    run: str
    func_dir: Path
    bold_path: Path
    confounds_path: Path
    confounds_json_path: Path
    aseg_path: Path
    mask_path: Path
    boldref_path: Path
    bold_available: bool = False
    confounds_available: bool = False
    aseg_available: bool = False
    mask_available: bool = False
    boldref_available: bool = False
    tr: float = 1.49  # default; overridden from JSON sidecar if present


def _is_available(path: Path) -> bool:
    """Return True if path resolves to a real, non-empty file."""
    try:
        real = path.resolve()
        return real.exists() and real.stat().st_size > 0
    except (OSError, PermissionError):
        return False


def _find_file(func_dir: Path, pattern: str) -> Path:
    """Return first match of glob pattern in func_dir, or a non-existent placeholder."""
    matches = sorted(func_dir.glob(pattern))
    if matches:
        return matches[0]
    # Return a predictable non-existent path so callers can still check availability
    return func_dir / f"MISSING_{pattern}"


def _get_tr(bold_json_path: Path) -> float:
    """Read RepetitionTime from BOLD sidecar JSON."""
    try:
        import json
        with open(bold_json_path) as f:
            meta = json.load(f)
        return float(meta.get("RepetitionTime", 1.49))
    except Exception:
        return 1.49


def discover_runs(
    fmriprep_dir: str | Path,
    subjects: Optional[List[str]] = None,
    space: str = "MNI152NLin2009cAsym",
) -> List[RunInfo]:
    """
    Walk the fmriprep directory and return one RunInfo per functional run.

    Parameters
    ----------
    fmriprep_dir:
        Root of the mario.fmriprep dataset.
    subjects:
        Optional list of subject IDs to restrict to (e.g. ['sub-01', 'sub-02']).
        If None, all subjects are discovered.
    space:
        Template space string used in file names.

    Returns
    -------
    List of RunInfo objects sorted by (subject, session, run).
    """
    root = Path(fmriprep_dir)
    runs: List[RunInfo] = []

    sub_dirs = sorted(d for d in root.iterdir() if d.is_dir() and re.match(r"sub-\d+", d.name))

    for sub_dir in sub_dirs:
        subject = sub_dir.name
        if subjects and subject not in subjects:
            continue

        ses_dirs = sorted(
            d for d in sub_dir.iterdir() if d.is_dir() and re.match(r"ses-\d+", d.name)
        )

        for ses_dir in ses_dirs:
            session = ses_dir.name
            func_dir = ses_dir / "func"
            if not func_dir.is_dir():
                continue

            # Find all runs by looking for confounds JSON files (always present)
            run_keys = set()
            for f in func_dir.glob(f"{subject}_{session}_task-mario_run-*_desc-confounds_timeseries.json"):
                m = re.search(r"run-(\d+)", f.name)
                if m:
                    run_keys.add(m.group(1))

            for run_id in sorted(run_keys):
                run_label = f"run-{run_id}"
                prefix = f"{subject}_{session}_task-mario_{run_label}"

                bold_path = func_dir / f"{prefix}_space-{space}_desc-preproc_bold.nii.gz"
                confounds_path = func_dir / f"{prefix}_desc-confounds_timeseries.tsv"
                confounds_json_path = func_dir / f"{prefix}_desc-confounds_timeseries.json"
                aseg_path = func_dir / f"{prefix}_space-{space}_desc-aseg_dseg.nii.gz"
                mask_path = func_dir / f"{prefix}_space-{space}_desc-brain_mask.nii.gz"
                boldref_path = func_dir / f"{prefix}_space-{space}_boldref.nii.gz"

                # Read TR from the BOLD JSON sidecar (always a real file, not annex)
                bold_json_path = func_dir / f"{prefix}_space-{space}_desc-preproc_bold.json"
                tr = _get_tr(bold_json_path) if bold_json_path.exists() else 1.49

                info = RunInfo(
                    subject=subject,
                    session=session,
                    run=run_label,
                    func_dir=func_dir,
                    bold_path=bold_path,
                    confounds_path=confounds_path,
                    confounds_json_path=confounds_json_path,
                    aseg_path=aseg_path,
                    mask_path=mask_path,
                    boldref_path=boldref_path,
                    bold_available=_is_available(bold_path),
                    confounds_available=_is_available(confounds_path),
                    aseg_available=_is_available(aseg_path),
                    mask_available=_is_available(mask_path),
                    boldref_available=_is_available(boldref_path),
                    tr=tr,
                )
                runs.append(info)

    return runs


def check_availability(runs: List[RunInfo]) -> pd.DataFrame:
    """
    Produce a summary DataFrame of file availability per run.

    Useful for --dry_run mode.
    """
    rows = []
    for r in runs:
        rows.append(
            {
                "subject": r.subject,
                "session": r.session,
                "run": r.run,
                "bold": r.bold_available,
                "confounds": r.confounds_available,
                "aseg": r.aseg_available,
                "mask": r.mask_available,
                "boldref": r.boldref_available,
            }
        )
    return pd.DataFrame(rows)


def print_availability_report(runs: List[RunInfo]) -> None:
    """Print a human-readable availability table."""
    df = check_availability(runs)
    print("\nCNeuromod Mario fMRIPrep Cerebellar QC — File Availability")
    print("=" * 65)

    cols = ["bold", "confounds", "aseg", "mask", "boldref"]
    summary = (
        df.groupby("subject")[cols]
        .agg(["sum", "count"])
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )

    print(f"\n{'Subject':<10} {'Sessions':>8} {'Runs':>6}", end="")
    for c in cols:
        print(f"  {c:>12}", end="")
    print()
    print("-" * 75)

    for subject, grp in df.groupby("subject"):
        n_ses = grp["session"].nunique()
        n_runs = len(grp)
        print(f"{subject:<10} {n_ses:>8} {n_runs:>6}", end="")
        for c in cols:
            n_avail = grp[c].sum()
            print(f"  {n_avail:>5}/{n_runs:<6}", end="")
        print()

    print("-" * 75)
    n_total = len(df)
    print(f"{'TOTAL':<10} {df['session'].nunique():>8} {n_total:>6}", end="")
    for c in cols:
        n_avail = df[c].sum()
        print(f"  {n_avail:>5}/{n_total:<6}", end="")
    print("\n")

    n_bold = df["bold"].sum()
    n_conf = df["confounds"].sum()
    print(f"INFO: {n_bold}/{n_total} runs have BOLD available (full mode).")
    print(f"INFO: {n_conf}/{n_total} runs have confounds TSV available (motion-only mode).")

    if n_bold < n_total:
        print("\nTo fetch BOLD data locally (example for sub-01/ses-001):")
        print("  datalad get sub-01/ses-001/func/*MNI*bold.nii.gz")
        print("  datalad get sub-01/ses-001/func/*confounds_timeseries.tsv")
        print("  datalad get sub-01/ses-001/func/*aseg*MNI*.nii.gz")
        print("  datalad get sub-01/ses-001/func/*brain_mask*MNI*.nii.gz")
        print("  datalad get sub-01/ses-001/func/*boldref*MNI*.nii.gz")
        print("\nUse --no_bold for confounds-only metrics without downloading BOLD.")
