"""
HTML report builder.

Assembles all sections into a single self-contained HTML file.
"""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.io as pio

from qc.atlas import SUIT_LABEL_MAP
from qc.report.figures import (
    _plotly_html,
    make_summary_table,
    make_tsnr_lobule_bar,
    make_tsnr_violin,
    make_tsnr_session_heatmap,
    make_fd_violin,
    make_fd_session_heatmap,
    make_fd_session_trend,
    make_carpet_figure,
    make_coverage_bar,
    make_dropout_bar,
    make_acompcor_variance_curve,
    make_csf_correlation_bar,
    make_ar1_heatmap,
    make_interlobule_corr_heatmap,
    make_signal_drift_scatter,
    make_motor_cereb_corr_bar,
    make_usable_volumes_bar,
    make_yeo_cereb_scatter,
    make_yeo_lobule_corr_matrix,
)
from qc.report.embed import tsnr_slices_to_html, tsnr_interactive_viewer, atlas_interactive_viewer


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CNeuromod — Cerebellar QC Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 1600px;
      margin: auto;
      padding: 20px;
      background: #f9f9f9;
      color: #2d3436;
    }}
    h1 {{ color: #2d3436; border-bottom: 3px solid #636EFA; padding-bottom: 10px; }}
    h2 {{ color: #636EFA; margin-top: 40px; border-left: 5px solid #636EFA; padding-left: 12px; }}
    h3 {{ color: #555; }}
    .section {{
      background: white;
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 32px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .subsection {{ margin-bottom: 28px; }}
    .meta {{ color: #636e72; font-size: 0.9em; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th {{ background: #636EFA; color: white; padding: 10px 14px; text-align: left; }}
    td {{ border: 1px solid #dfe6e9; padding: 8px 14px; }}
    tr:nth-child(even) {{ background: #f0f3ff; }}
    .warning {{ color: #d63031; font-weight: bold; }}
    .ok {{ color: #00b894; }}
    .info {{ color: #0984e3; }}
    .note {{
      background: #ffeaa7;
      border-left: 4px solid #fdcb6e;
      padding: 10px 14px;
      border-radius: 4px;
      margin-bottom: 16px;
      font-size: 0.9em;
    }}
  </style>
</head>
<body>
  <h1>CNeuroMod — Cerebellar QC Report</h1>
  <p class="meta">
    Generated: {timestamp} &nbsp;|&nbsp;
    fMRIPrep 20.2.5 &nbsp;|&nbsp;
    Space: MNI152NLin2009cAsym &nbsp;|&nbsp;
    Atlas: SUIT Anatom (34 regions) + FreeSurfer aseg
  </p>
  <div class="note">
    <strong>Note:</strong> This report covers all available subjects.
    Metrics are computed per run and aggregated per session/subject.
    N/A or missing values indicate files were unavailable (datalad not fetched locally).
  </div>

  {section_0}
  {section_1}
  {section_2}
  {section_3}
  {section_4}
  {section_5}
  {section_6}
  {section_7}

</body>
</html>
"""

_SECTION_TEMPLATE = """\
<div class="section" id="{section_id}">
  <h2>{title}</h2>
  {content}
</div>
"""


def _section(section_id: str, title: str, content: str) -> str:
    return _SECTION_TEMPLATE.format(section_id=section_id, title=title, content=content)


def _subsection(title: str, content: str) -> str:
    return f'<div class="subsection"><h3>{title}</h3>{content}</div>'


def generate_html_report(
    dfs: Dict[str, pd.DataFrame],
    run_results: List[dict],
    tsnr_imgs: Dict[str, object],
    output_path: str | Path,
    atlas_img: Optional[object] = None,
) -> None:
    """
    Assemble all sections into a single self-contained HTML report.

    Parameters
    ----------
    dfs:
        Dict from aggregate_results: keys 'runs', 'sessions', 'subjects'.
    run_results:
        Raw list of per-run result dicts (contains _tsnr_img, _carpet, etc.).
    tsnr_imgs:
        Dict: {subject: tsnr_img} — one representative tSNR image per subject
        (e.g., first available run).
    output_path:
        Where to write the HTML file.
    """
    df_runs = dfs.get("runs", pd.DataFrame())
    df_sessions = dfs.get("sessions", pd.DataFrame())
    df_subjects = dfs.get("subjects", pd.DataFrame())

    suit_label_names = list(SUIT_LABEL_MAP.values())

    # Track whether we've emitted the Plotly CDN script yet
    plotlyjs_emitted = [False]

    def _fig_html(fig) -> str:
        include_js = not plotlyjs_emitted[0]
        if include_js:
            plotlyjs_emitted[0] = True
        return _plotly_html(fig, include_plotlyjs=include_js)

    def _safe_fig(factory, *args, **kwargs) -> str:
        try:
            fig = factory(*args, **kwargs)
            return _fig_html(fig)
        except Exception as e:
            return f'<p class="warning">[Figure unavailable: {e}]</p>'

    # -----------------------------------------------------------------------
    # Section 0 — Summary table
    # -----------------------------------------------------------------------
    s0_content = ""
    if not df_subjects.empty and not df_runs.empty:
        s0_content += _safe_fig(make_summary_table, df_subjects, df_runs)
    else:
        s0_content += '<p class="warning">No data available for summary table.</p>'

    section_0 = _section("sec-0", "0. Executive Summary", s0_content)

    # -----------------------------------------------------------------------
    # Section 1 — tSNR
    # -----------------------------------------------------------------------
    s1_parts = []

    # Atlas reference viewer (shown first so it can be consulted alongside tSNR maps)
    if atlas_img is not None:
        s1_parts.append(_subsection(
            "SUIT Atlas Reference — interactive (click to navigate, compare with tSNR maps below)",
            atlas_interactive_viewer(atlas_img, label_map=SUIT_LABEL_MAP),
        ))

    # Compute a global vmax across all subjects so colour scales are comparable
    global_tsnr_vmax = None
    if tsnr_imgs:
        percentiles = []
        for tsnr_img in tsnr_imgs.values():
            if tsnr_img is None:
                continue
            d = tsnr_img.get_fdata()
            valid = d[np.isfinite(d) & (d > 0)]
            if len(valid) > 0:
                percentiles.append(float(np.percentile(valid, 97)))
        if percentiles:
            global_tsnr_vmax = float(np.percentile(percentiles, 95))

    # Interactive 3D tSNR viewers (one per subject, nilearn view_img)
    if tsnr_imgs:
        viewer_html = ""
        for subj, tsnr_img in sorted(tsnr_imgs.items()):
            if tsnr_img is not None:
                viewer_html += f"<h4>{subj}</h4>"
                viewer_html += tsnr_interactive_viewer(tsnr_img, subj, vmax=global_tsnr_vmax)
        if viewer_html:
            s1_parts.append(_subsection(
                "tSNR — Interactive 3D Viewer (click to navigate, hover for values)",
                viewer_html,
            ))

    # Static axial slice mosaics (overview, cerebellar focus)
    if tsnr_imgs:
        slice_html = ""
        for subj, tsnr_img in sorted(tsnr_imgs.items()):
            if tsnr_img is not None:
                slice_html += f"<h4>{subj}</h4>"
                slice_html += tsnr_slices_to_html(tsnr_img, subj, vmax=global_tsnr_vmax)
        if slice_html:
            s1_parts.append(_subsection("tSNR Maps (cerebellar axial slices, z = -50 to -20 mm)", slice_html))

    if not df_runs.empty:
        s1_parts.append(_subsection(
            "Mean tSNR per SUIT Lobule",
            _safe_fig(make_tsnr_lobule_bar, df_runs, suit_label_names),
        ))
        s1_parts.append(_subsection(
            "Cerebellar tSNR Distribution (all runs)",
            _safe_fig(make_tsnr_violin, df_runs),
        ))
        s1_parts.append(_subsection(
            "Cerebellar tSNR per Session × Run",
            _safe_fig(make_tsnr_session_heatmap, df_runs),
        ))

    section_1 = _section("sec-1", "1. Temporal SNR (tSNR)", "\n".join(s1_parts))

    # -----------------------------------------------------------------------
    # Section 2 — Motion
    # -----------------------------------------------------------------------
    s2_parts = []

    if not df_runs.empty:
        s2_parts.append(_subsection(
            "Framewise Displacement Distribution",
            _safe_fig(make_fd_violin, df_runs),
        ))
        s2_parts.append(_subsection(
            "% High-Motion Volumes per Session × Run",
            _safe_fig(make_fd_session_heatmap, df_runs),
        ))

    if not df_sessions.empty:
        s2_parts.append(_subsection(
            "Session-Level Mean FD Trend",
            _safe_fig(make_fd_session_trend, df_sessions),
        ))

    # Carpet plots (worst and best run per subject)
    carpet_html = ""
    subjects_done = set()
    for r in run_results:
        subj = r["subject"]
        carpet = r.get("_carpet")
        if carpet is None or carpet.get("n_voxels", 0) == 0:
            continue
        if subj in subjects_done:
            continue
        fd_series = r.get("_fd_series")
        run_label = f"{subj} {r['session']} {r['run']}"
        try:
            fig = make_carpet_figure(carpet, fd_series, run_label)
            carpet_html += f"<h4>{run_label}</h4>"
            carpet_html += _fig_html(fig)
        except Exception as e:
            carpet_html += f'<p class="warning">[Carpet plot error for {run_label}: {e}]</p>'
        subjects_done.add(subj)

    if carpet_html:
        s2_parts.append(_subsection("Cerebellar Carpet Plots (representative runs)", carpet_html))
    else:
        s2_parts.append('<p class="info">Carpet plots require BOLD data. Run without --no_bold flag.</p>')

    section_2 = _section("sec-2", "2. Head Motion", "\n".join(s2_parts))

    # -----------------------------------------------------------------------
    # Section 3 — Coverage and dropout
    # -----------------------------------------------------------------------
    s3_parts = []

    if not df_runs.empty:
        s3_parts.append(_subsection(
            "Brain Mask Coverage per SUIT Lobule",
            _safe_fig(make_coverage_bar, df_runs, suit_label_names),
        ))
        s3_parts.append(_subsection(
            "Signal Dropout per SUIT Lobule (from boldref)",
            _safe_fig(make_dropout_bar, df_runs, suit_label_names),
        ))

    section_3 = _section("sec-3", "3. Brain Mask Coverage & Signal Dropout", "\n".join(s3_parts))

    # -----------------------------------------------------------------------
    # Section 4 — Noise
    # -----------------------------------------------------------------------
    s4_parts = []

    if not df_runs.empty:
        s4_parts.append(_subsection(
            "aCompCor: Components to Reach 50% / 80% Variance",
            _safe_fig(make_acompcor_variance_curve, df_runs),
        ))
        s4_parts.append(_subsection(
            "4th Ventricle–Cerebellar Cortex Signal Correlation (CSF contamination)",
            _safe_fig(make_csf_correlation_bar, df_runs),
        ))

    section_4 = _section("sec-4", "4. Noise & Physiological Contamination", "\n".join(s4_parts))

    # -----------------------------------------------------------------------
    # Section 5 — Usable data
    # -----------------------------------------------------------------------
    s5_parts = []

    if not df_runs.empty:
        s5_parts.append(_subsection(
            "% Usable Volumes After Scrubbing per Session",
            _safe_fig(make_usable_volumes_bar, df_runs),
        ))

        # Summary table of total usable TRs
        if not df_subjects.empty and "n_usable" in df_subjects.columns and "tr" in df_runs.columns:
            rows = []
            for subj in sorted(df_subjects["subject"]):
                subj_runs = df_runs[df_runs["subject"] == subj]
                total_trs = float(subj_runs["n_trs"].sum(skipna=True)) if "n_trs" in subj_runs else float("nan")
                total_usable = float(subj_runs["n_usable"].sum(skipna=True)) if "n_usable" in subj_runs else float("nan")
                tr_val = float(subj_runs["tr"].mean(skipna=True)) if "tr" in subj_runs else 1.49
                pct = 100.0 * total_usable / total_trs if total_trs > 0 else float("nan")
                usable_min = total_usable * tr_val / 60.0
                rows.append({
                    "Subject": subj,
                    "Total TRs": f"{int(total_trs):,}" if np.isfinite(total_trs) else "N/A",
                    "Usable TRs": f"{int(total_usable):,}" if np.isfinite(total_usable) else "N/A",
                    "% Usable": f"{pct:.1f}%" if np.isfinite(pct) else "N/A",
                    "Usable time (min)": f"{usable_min:.1f}" if np.isfinite(usable_min) else "N/A",
                })
            tbl_html = pd.DataFrame(rows).to_html(index=False, classes="", border=0)
            s5_parts.append(_subsection("Usable Data Summary Table", tbl_html))

    section_5 = _section("sec-5", "5. Usable Data After Scrubbing", "\n".join(s5_parts))

    # -----------------------------------------------------------------------
    # Section 6 — Advanced metrics
    # -----------------------------------------------------------------------
    s6_parts = []
    s6_parts.append(
        '<div class="note">These metrics are particularly relevant for interpreting '
        'cerebellar activity during gameplay. They characterize the noise structure '
        'and shared variance that will affect GLM analysis.</div>'
    )

    if not df_runs.empty:
        s6_parts.append(_subsection(
            "Temporal Autocorrelation AR(1) — Mean Cerebellar Signal",
            _safe_fig(make_ar1_heatmap, df_runs),
        ))

    # Inter-lobule correlation matrices (one per subject, from first available run)
    interlobule_html = ""
    subj_interlobule_done = set()
    for r in run_results:
        subj = r["subject"]
        corr = r.get("_interlobule_corr")
        names = r.get("_interlobule_names")
        if corr is None or names is None or subj in subj_interlobule_done:
            continue
        try:
            fig = make_interlobule_corr_heatmap(corr, names, subj)
            interlobule_html += f"<h4>{subj}</h4>"
            interlobule_html += _fig_html(fig)
            subj_interlobule_done.add(subj)
        except Exception as e:
            interlobule_html += f'<p class="warning">[Inter-lobule corr error for {subj}: {e}]</p>'

    if interlobule_html:
        s6_parts.append(_subsection(
            "Inter-Lobule Correlation Matrix (raw signal, no denoising)",
            interlobule_html,
        ))
    else:
        s6_parts.append('<p class="info">Inter-lobule correlation requires BOLD data.</p>')

    if not df_sessions.empty:
        s6_parts.append(_subsection(
            "Session-Level Cerebellar Signal Drift (%/min)",
            _safe_fig(make_signal_drift_scatter, df_sessions),
        ))

    if not df_runs.empty:
        s6_parts.append(_subsection(
            "Cerebellum–Motor Cortex Baseline Correlation",
            _safe_fig(make_motor_cereb_corr_bar, df_runs),
        ))

    section_6 = _section("sec-6", "6. Advanced Metrics for Gameplay Analysis", "\n".join(s6_parts))

    # -----------------------------------------------------------------------
    # Section 7 — Cortex–Cerebellum tSNR comparison
    # -----------------------------------------------------------------------
    s7_parts = []
    s7_parts.append(
        '<div class="note">tSNR is extracted per Yeo 7-network restricted to cerebral cortex GM '
        '(FreeSurfer aseg labels 3+42), matching the cerebellar GM masking approach. '
        'Correlations across sessions reflect shared scan-quality fluctuation, '
        '<strong>not functional connectivity</strong>.</div>'
    )

    if not df_sessions.empty:
        s7_parts.append(_subsection(
            "Yeo Network vs Whole-Cerebellum tSNR (session-level dots)",
            _safe_fig(make_yeo_cereb_scatter, df_sessions),
        ))
        s7_parts.append(_subsection(
            "Yeo Network × SUIT Lobule tSNR Correlation Matrix (per subject, across sessions)",
            _safe_fig(make_yeo_lobule_corr_matrix, df_sessions, suit_label_names),
        ))
    else:
        s7_parts.append('<p class="info">Yeo–Cerebellum plots require BOLD data.</p>')

    section_7 = _section("sec-7", "7. Cortex–Cerebellum tSNR Comparison (Yeo 7 Networks)", "\n".join(s7_parts))

    # -----------------------------------------------------------------------
    # Assemble final HTML
    # -----------------------------------------------------------------------
    html = _HTML_TEMPLATE.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        section_0=section_0,
        section_1=section_1,
        section_2=section_2,
        section_3=section_3,
        section_4=section_4,
        section_5=section_5,
        section_6=section_6,
        section_7=section_7,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
