# CNeuroMod — Cerebellar QC Pipeline

Quality control pipeline for fMRIPrep-preprocessed datasets with a focus on cerebellar signal. Produces a single interactive HTML report covering tSNR, head motion, brain mask coverage, physiological noise, and advanced cerebellar metrics.

---

## Installation

```bash
git clone <repo-url>
cd cneuromod_cerebellum_qc
pip install -e .
```

This installs the package in editable mode and registers three CLI commands: `cereb-qc`, `download-suit-atlas`, and `plot-atlas`.

To also install development dependencies (pytest):

```bash
pip install -e ".[dev]"
```

---

## Quick start

```bash
# 1. Download the SUIT cerebellar atlas (once)
download-suit-atlas --output_dir data/

# 2. Check file availability before committing to a full run
cereb-qc --fmriprep_dir /path/to/dataset.fmriprep --dry_run

# 3. Full run
cereb-qc --fmriprep_dir /path/to/dataset.fmriprep --n_jobs 8

# 4. (Optional) Static atlas figure
plot-atlas --atlas data/suit_atl-Anatom_space-MNI_dseg.nii.gz
```

Reports and CSVs are written to `output/{dataset}/` (derived from the fmriprep directory name).

---

## CLI reference

### `download-suit-atlas`

Downloads the SUIT Anatom cerebellar atlas (Diedrichsen 2009, 34 regions) from the [Diedrichsen Lab GitHub](https://github.com/DiedrichsenLab/cerebellar_atlases). Saves a NIfTI and the authoritative label TSV.

```
download-suit-atlas --output_dir <dir> [--no_verify]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--output_dir` | yes | — | Directory where atlas files are saved |
| `--no_verify` | no | false | Skip post-download integrity check |

**Output:**

| File | Description |
|---|---|
| `suit_atl-Anatom_space-MNI_dseg.nii.gz` | Discrete label atlas in MNI space |
| `suit_atl-Anatom_space-MNI_dseg.tsv` | Authoritative region names and colours |

---

### `cereb-qc`

Main entry point. Discovers all subjects/sessions/runs in an fMRIPrep directory, computes QC metrics, and writes a self-contained HTML report.

```
cereb-qc --fmriprep_dir <path> [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--fmriprep_dir` | yes | — | Root of the fmriprep dataset |
| `--output_dir` | no | `output` | Parent output directory; results go to `{output_dir}/{dataset}/` |
| `--suit_atlas` | no | `data/suit_atl-Anatom_space-MNI_dseg.nii.gz` | SUIT atlas NIfTI (run `download-suit-atlas` first) |
| `--subjects` | no | all | Restrict to specific subjects, e.g. `sub-01 sub-02` |
| `--fd_threshold` | no | `0.5` | Framewise displacement scrubbing threshold (mm) |
| `--dvars_threshold` | no | `1.5` | Standardised DVARS scrubbing threshold |
| `--n_jobs` | no | `1` | Parallel workers. Each BOLD worker uses ~4 GB RAM; recommend 4–8 on HPC |
| `--no_bold` | no | false | Confounds-only mode — skip BOLD loading (~5 min for all runs) |
| `--dry_run` | no | false | Print file availability table and exit without computing |
| `--report_name` | no | `cerebellar_qc_report.html` | Name of the output HTML file |

**Modes:**

```bash
# Dry run — check which files are locally available (no computation)
cereb-qc --fmriprep_dir /path/to/ds.fmriprep --dry_run

# Fast mode — FD, DVARS, aCompCor only; no BOLD files needed
cereb-qc --fmriprep_dir /path/to/ds.fmriprep --no_bold

# Full mode — all metrics including tSNR, coverage, noise
cereb-qc \
    --fmriprep_dir /path/to/ds.fmriprep \
    --output_dir /scratch/results \
    --n_jobs 8

# Single subject
cereb-qc --fmriprep_dir /path/to/ds.fmriprep --subjects sub-01 --n_jobs 2
```

**Output (written to `{output_dir}/{dataset}/`):**

| File | Description |
|---|---|
| `cerebellar_qc_report.html` | Self-contained interactive HTML report |
| `qc_runs.csv` | One row per run — all metrics |
| `qc_sessions.csv` | Session-level aggregates |
| `qc_subjects.csv` | Subject-level aggregates |

**Report sections:**

| Section | Contents |
|---|---|
| tSNR | Interactive SUIT atlas reference viewer; per-lobule tSNR bar/violin/heatmap; interactive 3D tSNR viewer per subject |
| Motion | FD and DVARS distributions; session-level FD trend; carpet plots |
| Coverage & dropout | Brain mask coverage per lobule; signal dropout from boldref |
| Noise | aCompCor variance curve; 4th-ventricle CSF correlation; AR(1) autocorrelation |
| Usable data | % usable volumes after scrubbing; total usable minutes per subject |
| Advanced | Signal drift; cerebellum–motor cortex correlation; inter-lobule correlation matrix |

---

### `plot-atlas`

Generates a static multi-panel figure of the SUIT atlas overlaid on the MNI152 template, with official region colours and a labelled legend. The TSV must be present alongside the NIfTI (downloaded automatically by `download-suit-atlas`).

```
plot-atlas --atlas <path> [--output <path>] [--dpi <int>]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--atlas` | yes | — | Path to `suit_atl-Anatom_space-MNI_dseg.nii.gz` |
| `--output` | no | `atlas_view.png` | Output image path |
| `--dpi` | no | `150` | Figure resolution |

---

## Dataset requirements

The pipeline expects an fMRIPrep output directory structured as:

```
{subject}/{session}/func/
    {sub}_{ses}_task-{task}_run-{N}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    {sub}_{ses}_task-{task}_run-{N}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    {sub}_{ses}_task-{task}_run-{N}_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz
    {sub}_{ses}_task-{task}_run-{N}_space-MNI152NLin2009cAsym_boldref.nii.gz
    {sub}_{ses}_task-{task}_run-{N}_desc-confounds_timeseries.tsv
    {sub}_{ses}_task-{task}_run-{N}_desc-confounds_timeseries.json
```

The task label is inferred automatically from filenames. BOLD files may be datalad/git-annex symlinks; only confounds JSONs are required locally for `--dry_run` and `--no_bold` modes.

---

## Tests

```bash
pytest tests/ -v
```

Tests use synthetic NIfTI data and do not require any real dataset or atlas files.

---

## Atlas

The SUIT Anatom parcellation (Diedrichsen 2009) labels 34 cerebellar regions:

- **Cortical lobules (left / vermis / right):** I–IV, V, VI, CrusI, CrusII, VIIb, VIIIa, VIIIb, IX, X
- **Deep nuclei (left / right):** Dentate, Interposed, Fastigial

The atlas is provided in MNI space. fMRIPrep outputs use MNI152NLin2009cAsym; the template mismatch is sub-voxel for most of the cerebellum and acceptable for QC purposes.