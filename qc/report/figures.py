"""
Reusable Plotly figure factories for the QC report.

All figures use a consistent subject color palette.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Subject color palette (fixed, consistent across all figures)
# ---------------------------------------------------------------------------
SUBJECT_COLORS = {
    "sub-01": "#636EFA",  # blue
    "sub-02": "#EF553B",  # red
    "sub-03": "#00CC96",  # green
    "sub-05": "#AB63FA",  # purple
    "sub-06": "#FFA15A",  # orange
}

_DEFAULT_COLOR = "#AAAAAA"


def subject_color(subject: str) -> str:
    return SUBJECT_COLORS.get(subject, _DEFAULT_COLOR)


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#RRGGBB' to 'rgba(R,G,B,alpha)'. Plotly rejects 8-char hex."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plotly_html(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    import plotly.io as pio
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={"displayModeBar": True, "scrollZoom": True},
    )


def _fig_title(text: str, fontsize: int = 16) -> dict:
    return {"text": text, "x": 0.5, "xanchor": "center", "font": {"size": fontsize}}


# ---------------------------------------------------------------------------
# Section 0 — Summary table
# ---------------------------------------------------------------------------

def make_summary_table(df_subjects: pd.DataFrame, df_runs: pd.DataFrame) -> go.Figure:
    """Executive summary table: one row per subject."""
    subjects = df_subjects["subject"].tolist()

    # Count sessions and runs per subject
    ses_counts = df_runs.groupby("subject")["session"].nunique()
    run_counts = df_runs.groupby("subject").size()

    rows = []
    for subj in subjects:
        row = df_subjects[df_subjects["subject"] == subj].iloc[0]
        rows.append(
            {
                "Subject": subj,
                "Sessions": int(ses_counts.get(subj, 0)),
                "Runs": int(run_counts.get(subj, 0)),
                "Cereb tSNR": f"{row.get('cereb_mean', float('nan')):.1f}" if pd.notna(row.get("cereb_mean")) else "N/A",
                "Cereb/WB ratio": f"{row.get('cereb_wb_ratio', float('nan')):.2f}" if pd.notna(row.get("cereb_wb_ratio")) else "N/A",
                "Mean FD (mm)": f"{row.get('mean_fd', float('nan')):.3f}" if pd.notna(row.get("mean_fd")) else "N/A",
                "% Usable": f"{row.get('pct_usable', float('nan')):.1f}%" if pd.notna(row.get("pct_usable")) else "N/A",
                "Mask Cov. Min": f"{row.get('suit_coverage_min', float('nan')):.0%}" if pd.notna(row.get("suit_coverage_min")) else "N/A",
                "AR1 Cereb": f"{row.get('ar1_cereb_mean', float('nan')):.2f}" if pd.notna(row.get("ar1_cereb_mean")) else "N/A",
            }
        )

    tbl = pd.DataFrame(rows)
    fig = go.Figure(
        data=go.Table(
            header=dict(
                values=list(tbl.columns),
                fill_color="#2d3436",
                font=dict(color="white", size=13),
                align="center",
            ),
            cells=dict(
                values=[tbl[c] for c in tbl.columns],
                fill_color=[
                    [_hex_to_rgba(subject_color(s)) for s in subjects]
                    for _ in tbl.columns
                ],
                font=dict(color="#2d3436", size=12),
                align="center",
                height=28,
            ),
        )
    )
    fig.update_layout(
        title=_fig_title("Executive Summary — All Subjects"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=max(250, 50 + 35 * len(subjects)),
    )
    return fig


# ---------------------------------------------------------------------------
# Section 1 — tSNR figures
# ---------------------------------------------------------------------------

def make_tsnr_lobule_bar(df_runs: pd.DataFrame, suit_label_names: List[str]) -> go.Figure:
    """Grouped bar chart: mean tSNR per SUIT lobule, grouped by subject."""
    subjects = sorted(df_runs["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        subj_df = df_runs[df_runs["subject"] == subj]
        vals = []
        for lname in suit_label_names:
            col = f"suit_{lname}"
            if col in subj_df.columns:
                vals.append(float(subj_df[col].mean(skipna=True)))
            else:
                vals.append(float("nan"))

        fig.add_trace(
            go.Bar(
                name=subj,
                x=suit_label_names,
                y=vals,
                marker_color=subject_color(subj),
                error_y=dict(
                    type="data",
                    array=[
                        float(subj_df[f"suit_{lname}"].std(skipna=True))
                        if f"suit_{lname}" in subj_df.columns else float("nan")
                        for lname in suit_label_names
                    ],
                    visible=True,
                ),
            )
        )

    fig.update_layout(
        barmode="group",
        title=_fig_title("Mean tSNR per SUIT Lobule (averaged over all sessions/runs)"),
        xaxis_title="SUIT Lobule",
        yaxis_title="tSNR",
        xaxis_tickangle=-45,
        legend_title="Subject",
        height=450,
        margin=dict(b=120),
    )
    return fig


def make_tsnr_violin(df_runs: pd.DataFrame) -> go.Figure:
    """Violin plot: tSNR distribution per subject (across all sessions/runs)."""
    subjects = sorted(df_runs["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        vals = df_runs[df_runs["subject"] == subj]["cereb_mean"].dropna().values
        if len(vals) == 0:
            continue
        fig.add_trace(
            go.Violin(
                y=vals,
                name=subj,
                box_visible=True,
                meanline_visible=True,
                fillcolor=subject_color(subj),
                line_color=subject_color(subj),
                opacity=0.7,
            )
        )

    fig.update_layout(
        title=_fig_title("Cerebellar tSNR Distribution per Subject (all runs)"),
        yaxis_title="Cerebellar tSNR",
        xaxis_title="Subject",
        showlegend=False,
        height=400,
    )
    return fig


def make_tsnr_session_heatmap(df_runs: pd.DataFrame) -> go.Figure:
    """Per-subject tSNR heatmap: session (y) × run (x)."""
    subjects = sorted(df_runs["subject"].unique())
    n_subj = len(subjects)
    fig = make_subplots(
        rows=1,
        cols=n_subj,
        subplot_titles=subjects,
        shared_yaxes=False,
    )

    for col_idx, subj in enumerate(subjects, start=1):
        subj_df = df_runs[df_runs["subject"] == subj].copy()
        sessions = sorted(subj_df["session"].unique(), key=lambda s: int(s.split("-")[1]))
        runs = sorted(subj_df["run"].unique(), key=lambda r: int(r.split("-")[1]))

        z = np.full((len(sessions), len(runs)), np.nan)
        for i, ses in enumerate(sessions):
            for j, run in enumerate(runs):
                match = subj_df[(subj_df["session"] == ses) & (subj_df["run"] == run)]
                if not match.empty and "cereb_mean" in match.columns:
                    z[i, j] = match["cereb_mean"].values[0]

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=runs,
                y=sessions,
                colorscale="Viridis",
                showscale=(col_idx == n_subj),
                colorbar=dict(title="tSNR", x=1.02) if col_idx == n_subj else None,
                zmin=0,
                hovertemplate="Session: %{y}<br>Run: %{x}<br>tSNR: %{z:.1f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=_fig_title("Cerebellar tSNR per Session × Run"),
        height=max(300, 20 * len(df_runs["session"].unique()) + 100),
    )
    return fig


# ---------------------------------------------------------------------------
# Section 2 — Motion figures
# ---------------------------------------------------------------------------

def make_fd_violin(df_runs: pd.DataFrame) -> go.Figure:
    """Violin: FD distribution per subject."""
    subjects = sorted(df_runs["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        vals = df_runs[df_runs["subject"] == subj]["mean_fd"].dropna().values
        if len(vals) == 0:
            continue
        fig.add_trace(
            go.Violin(
                y=vals,
                name=subj,
                box_visible=True,
                meanline_visible=True,
                fillcolor=subject_color(subj),
                line_color=subject_color(subj),
                opacity=0.7,
            )
        )

    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="FD=0.5mm threshold", annotation_position="top right")
    fig.update_layout(
        title=_fig_title("Mean Framewise Displacement per Run (distribution per subject)"),
        yaxis_title="Mean FD (mm)",
        showlegend=False,
        height=400,
    )
    return fig


def make_fd_session_heatmap(df_runs: pd.DataFrame) -> go.Figure:
    """Per-subject heatmap: % high-motion volumes per session × run."""
    subjects = sorted(df_runs["subject"].unique())
    n_subj = len(subjects)
    fig = make_subplots(rows=1, cols=n_subj, subplot_titles=subjects)

    for col_idx, subj in enumerate(subjects, start=1):
        subj_df = df_runs[df_runs["subject"] == subj].copy()
        sessions = sorted(subj_df["session"].unique(), key=lambda s: int(s.split("-")[1]))
        runs = sorted(subj_df["run"].unique(), key=lambda r: int(r.split("-")[1]))

        z = np.full((len(sessions), len(runs)), np.nan)
        for i, ses in enumerate(sessions):
            for j, run in enumerate(runs):
                match = subj_df[(subj_df["session"] == ses) & (subj_df["run"] == run)]
                if not match.empty and "pct_fd_above_threshold" in match.columns:
                    z[i, j] = match["pct_fd_above_threshold"].values[0]

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=runs,
                y=sessions,
                colorscale="Reds",
                showscale=(col_idx == n_subj),
                colorbar=dict(title="% vols", x=1.02) if col_idx == n_subj else None,
                zmin=0,
                zmax=50,
                hovertemplate="Session: %{y}<br>Run: %{x}<br>%% high-motion: %{z:.1f}%%<extra></extra>",
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        title=_fig_title("% High-Motion Volumes (FD > 0.5mm) per Session × Run"),
        height=max(300, 20 * df_runs["session"].nunique() + 100),
    )
    return fig


def make_fd_session_trend(df_sessions: pd.DataFrame) -> go.Figure:
    """Line plot: mean FD per session with regression trend per subject."""
    subjects = sorted(df_sessions["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        subj_df = df_sessions[df_sessions["subject"] == subj].sort_values("session_num")
        if "mean_fd" not in subj_df.columns or subj_df["mean_fd"].isna().all():
            continue
        x = subj_df["session_num"].values
        y = subj_df["mean_fd"].values
        color = subject_color(subj)

        # Data line
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines+markers",
                name=subj,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"{subj}<br>Session: %{{x}}<br>Mean FD: %{{y:.3f}} mm<extra></extra>",
            )
        )

        # Regression line
        valid = np.isfinite(y)
        if valid.sum() >= 3:
            from scipy.stats import linregress
            slope, intercept, r, p, _ = linregress(x[valid], y[valid])
            x_line = np.array([x[valid].min(), x[valid].max()])
            y_line = slope * x_line + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    name=f"{subj} trend",
                    line=dict(color=color, width=1, dash="dash"),
                    showlegend=False,
                    hovertemplate=f"{subj} trend (slope={slope:.4f} mm/session, p={p:.3f})<extra></extra>",
                )
            )

    fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                  annotation_text="0.5mm", annotation_position="right")
    fig.update_layout(
        title=_fig_title("Session-Level Mean FD Trend (motion over sessions)"),
        xaxis_title="Session index",
        yaxis_title="Mean FD (mm)",
        legend_title="Subject",
        height=400,
    )
    return fig


def make_carpet_figure(carpet_data: dict, fd_series: Optional[np.ndarray], run_label: str) -> go.Figure:
    """
    Carpet plot: cerebellar voxels × time with FD overlay above.

    Parameters
    ----------
    carpet_data:
        Dict from extract_carpet_data with keys 'data', 'voxel_labels', 'n_voxels'.
    fd_series:
        1D FD array aligned with the time axis.
    run_label:
        String label for the title (e.g., 'sub-01 ses-001 run-1').
    """
    data = carpet_data.get("data", np.zeros((0, 1)))
    voxel_labels = carpet_data.get("voxel_labels", [])
    n_voxels, n_trs = data.shape

    if n_voxels == 0:
        fig = go.Figure()
        fig.add_annotation(text="No cerebellar voxels available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Two rows: FD on top, carpet below
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.15, 0.85],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # FD trace
    if fd_series is not None and len(fd_series) == n_trs:
        fd_plot = np.nan_to_num(fd_series, nan=0.0)
        fig.add_trace(
            go.Scatter(
                x=list(range(n_trs)), y=fd_plot,
                fill="tozeroy",
                name="FD (mm)",
                line=dict(color="orange", width=1),
                hovertemplate="TR: %{x}<br>FD: %{y:.3f} mm<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=1)
        fig.update_yaxes(title_text="FD (mm)", row=1, col=1, range=[0, max(2, float(np.nanmax(fd_series)))])

    # Carpet heatmap
    # Build y-tick labels from unique lobule names at transitions
    tick_vals = []
    tick_texts = []
    prev_label = None
    for i, lbl in enumerate(voxel_labels):
        if lbl != prev_label:
            tick_vals.append(i)
            tick_texts.append(lbl)
            prev_label = lbl

    fig.add_trace(
        go.Heatmap(
            z=data,
            colorscale="RdBu",
            zmid=0,
            zmin=-3,
            zmax=3,
            showscale=True,
            colorbar=dict(title="z-score", thickness=12, len=0.8, y=0.4),
            hovertemplate="Voxel: %{y}<br>TR: %{x}<br>z: %{z:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(
        tickvals=tick_vals,
        ticktext=tick_texts,
        tickfont=dict(size=9),
        row=2, col=1,
        title_text="Cerebellar lobule",
    )
    fig.update_xaxes(title_text="Time (TR)", row=2, col=1)
    fig.update_layout(
        title=_fig_title(f"Cerebellar Carpet Plot — {run_label}"),
        height=500,
        margin=dict(l=120),
    )
    return fig


# ---------------------------------------------------------------------------
# Section 3 — Coverage and dropout
# ---------------------------------------------------------------------------

def make_coverage_bar(df_runs: pd.DataFrame, suit_label_names: List[str]) -> go.Figure:
    """Bar chart: % brain mask coverage per SUIT lobule per subject."""
    subjects = sorted(df_runs["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        subj_df = df_runs[df_runs["subject"] == subj]
        vals = []
        for lname in suit_label_names:
            col = f"suit_coverage_{lname}"
            if col in subj_df.columns:
                v = float(subj_df[col].mean(skipna=True))
                vals.append(100.0 * v if np.isfinite(v) else float("nan"))
            else:
                vals.append(float("nan"))

        fig.add_trace(
            go.Bar(
                name=subj,
                x=suit_label_names,
                y=vals,
                marker_color=subject_color(subj),
            )
        )

    fig.add_hline(y=80, line_dash="dash", line_color="orange",
                  annotation_text="80% threshold", annotation_position="right")
    fig.update_layout(
        barmode="group",
        title=_fig_title("Brain Mask Coverage per SUIT Lobule (%)"),
        xaxis_title="SUIT Lobule",
        yaxis_title="Coverage (%)",
        yaxis_range=[0, 105],
        xaxis_tickangle=-45,
        legend_title="Subject",
        height=450,
        margin=dict(b=120),
    )
    return fig


def make_dropout_bar(df_runs: pd.DataFrame, suit_label_names: List[str]) -> go.Figure:
    """Bar chart: relative EPI signal per SUIT lobule (vs. whole-brain mean)."""
    subjects = sorted(df_runs["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        subj_df = df_runs[df_runs["subject"] == subj]
        vals = []
        for lname in suit_label_names:
            col = f"dropout_{lname}"
            if col in subj_df.columns:
                vals.append(float(subj_df[col].mean(skipna=True)))
            else:
                vals.append(float("nan"))

        fig.add_trace(
            go.Bar(
                name=subj,
                x=suit_label_names,
                y=vals,
                marker_color=subject_color(subj),
            )
        )

    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Dropout threshold (50%)", annotation_position="right")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="Whole-brain mean", annotation_position="right")
    fig.update_layout(
        barmode="group",
        title=_fig_title("Relative EPI Signal per SUIT Lobule (boldref, vs. whole-brain mean)"),
        xaxis_title="SUIT Lobule",
        yaxis_title="Relative signal",
        xaxis_tickangle=-45,
        legend_title="Subject",
        height=450,
        margin=dict(b=120),
    )
    return fig


# ---------------------------------------------------------------------------
# Section 4 — Noise
# ---------------------------------------------------------------------------

def make_acompcor_variance_curve(df_runs: pd.DataFrame) -> go.Figure:
    """
    Placeholder for cumulative variance curve — requires per-run cumvar arrays.

    This figure visualizes n_for_50pct and n_for_80pct as a bar chart instead,
    since the full curve data is not stored in the aggregated DataFrame.
    """
    subjects = sorted(df_runs["subject"].unique())
    mask_types = ["CSF", "WM", "combined"]
    colors_by_mask = {"CSF": "#74b9ff", "WM": "#fd79a8", "combined": "#55efc4"}

    fig = make_subplots(
        rows=1, cols=len(mask_types),
        subplot_titles=[f"aCompCor — {m}" for m in mask_types],
    )

    for col_idx, mask_type in enumerate(mask_types, start=1):
        col_50 = f"acompcor_{mask_type}_n_for_50pct"
        col_n = f"acompcor_{mask_type}_n_retained"

        for subj in subjects:
            subj_df = df_runs[df_runs["subject"] == subj]
            val_50 = float(subj_df[col_50].mean(skipna=True)) if col_50 in subj_df.columns else float("nan")
            val_n = float(subj_df[col_n].mean(skipna=True)) if col_n in subj_df.columns else float("nan")

            fig.add_trace(
                go.Bar(
                    name=subj,
                    x=[subj],
                    y=[val_50],
                    marker_color=subject_color(subj),
                    opacity=0.7,
                    showlegend=(col_idx == 1),
                    hovertemplate=f"{subj} {mask_type}: n_for_50pct=%{{y:.0f}} / {val_n:.0f} total<extra></extra>",
                ),
                row=1, col=col_idx,
            )

    fig.update_yaxes(title_text="N components", row=1, col=1)
    fig.update_layout(
        title=_fig_title("aCompCor: Components needed to reach 50% cumulative variance"),
        height=400,
        barmode="group",
    )
    return fig


def make_csf_correlation_bar(df_runs: pd.DataFrame) -> go.Figure:
    """Bar chart: 4th ventricle–cerebellum Pearson r per subject."""
    subjects = sorted(df_runs["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        subj_df = df_runs[df_runs["subject"] == subj]
        for side, col in [("Left", "csf_cereb_corr_L"), ("Right", "csf_cereb_corr_R")]:
            if col not in subj_df.columns:
                continue
            val = float(subj_df[col].mean(skipna=True))
            fig.add_trace(
                go.Bar(
                    name=f"{subj} ({side})",
                    x=[f"{subj}\n({side})"],
                    y=[val],
                    marker_color=subject_color(subj),
                    opacity=0.9 if side == "Left" else 0.5,
                    hovertemplate=f"{subj} {side}: r=%{{y:.3f}}<extra></extra>",
                )
            )

    fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                  annotation_text="Warning threshold (r=0.3)", annotation_position="right")
    fig.update_layout(
        title=_fig_title("4th Ventricle–Cerebellar Cortex Signal Correlation (CSF contamination check)"),
        yaxis_title="Pearson r",
        xaxis_title="Subject (hemisphere)",
        showlegend=True,
        height=400,
    )
    return fig


def make_ar1_heatmap(df_runs: pd.DataFrame) -> go.Figure:
    """Heatmap: AR(1) coefficient per session × run, faceted by subject."""
    subjects = sorted(df_runs["subject"].unique())
    n_subj = len(subjects)
    fig = make_subplots(rows=1, cols=n_subj, subplot_titles=subjects)

    for col_idx, subj in enumerate(subjects, start=1):
        subj_df = df_runs[df_runs["subject"] == subj].copy()
        sessions = sorted(subj_df["session"].unique(), key=lambda s: int(s.split("-")[1]))
        runs = sorted(subj_df["run"].unique(), key=lambda r: int(r.split("-")[1]))

        z = np.full((len(sessions), len(runs)), np.nan)
        for i, ses in enumerate(sessions):
            for j, run in enumerate(runs):
                match = subj_df[(subj_df["session"] == ses) & (subj_df["run"] == run)]
                if not match.empty and "ar1_cereb_mean" in match.columns:
                    z[i, j] = match["ar1_cereb_mean"].values[0]

        fig.add_trace(
            go.Heatmap(
                z=z, x=runs, y=sessions,
                colorscale="RdYlGn_r",
                showscale=(col_idx == n_subj),
                zmin=0, zmax=0.8,
                colorbar=dict(title="AR(1)", x=1.02) if col_idx == n_subj else None,
                hovertemplate="Session: %{y}<br>Run: %{x}<br>AR(1): %{z:.3f}<extra></extra>",
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        title=_fig_title("Temporal Autocorrelation AR(1) — Mean Cerebellar Cortex"),
        height=max(300, 20 * df_runs["session"].nunique() + 100),
    )
    return fig


def make_interlobule_corr_heatmap(corr_matrix: np.ndarray, lobule_names: List[str], subject: str) -> go.Figure:
    """Inter-lobule Pearson correlation matrix for a single subject."""
    fig = go.Figure(
        go.Heatmap(
            z=corr_matrix,
            x=lobule_names,
            y=lobule_names,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Pearson r"),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>r: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=_fig_title(f"Inter-Lobule Correlation Matrix — {subject} (raw, no denoising)"),
        xaxis_tickangle=-45,
        height=520,
        margin=dict(b=120, l=120),
    )
    return fig


def make_signal_drift_scatter(df_sessions: pd.DataFrame) -> go.Figure:
    """Scatter: signal drift slope per session per subject."""
    subjects = sorted(df_sessions["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        subj_df = df_sessions[df_sessions["subject"] == subj].sort_values("session_num")
        if "signal_drift_pct" not in subj_df.columns:
            continue
        x = subj_df["session_num"].values
        y = subj_df["signal_drift_pct"].fillna(0).values
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines+markers",
                name=subj,
                line=dict(color=subject_color(subj), width=2),
                marker=dict(size=5),
                hovertemplate=f"{subj}<br>Session: %{{x}}<br>Drift: %{{y:.2f}}%/min<extra></extra>",
            )
        )

    fig.add_hline(y=2, line_dash="dash", line_color="orange",
                  annotation_text="+2%/min warning", annotation_position="right")
    fig.add_hline(y=-2, line_dash="dash", line_color="orange")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=_fig_title("Session-Level Cerebellar Signal Drift (%/min)"),
        xaxis_title="Session index",
        yaxis_title="Drift (%/min)",
        legend_title="Subject",
        height=400,
    )
    return fig


def make_motor_cereb_corr_bar(df_runs: pd.DataFrame) -> go.Figure:
    """Bar chart: cerebellum–motor cortex baseline correlation per subject."""
    subjects = sorted(df_runs["subject"].unique())
    fig = go.Figure()

    vals = []
    errs = []
    for subj in subjects:
        subj_df = df_runs[df_runs["subject"] == subj]
        if "motor_cereb_corr" not in subj_df.columns:
            vals.append(float("nan"))
            errs.append(0.0)
            continue
        col = subj_df["motor_cereb_corr"].dropna()
        vals.append(float(col.mean()) if len(col) > 0 else float("nan"))
        errs.append(float(col.std()) if len(col) > 1 else 0.0)

    fig.add_trace(
        go.Bar(
            x=subjects,
            y=vals,
            error_y=dict(type="data", array=errs, visible=True),
            marker_color=[subject_color(s) for s in subjects],
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Warning threshold (r=0.5)", annotation_position="right")
    fig.update_layout(
        title=_fig_title("Cerebellum–Motor Cortex Baseline Correlation (raw signal, no denoising)"),
        xaxis_title="Subject",
        yaxis_title="Pearson r",
        showlegend=False,
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Section 5 — Usable data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Section 7 — Cortex–Cerebellum tSNR comparison
# ---------------------------------------------------------------------------

YEO7_NAMES = ["Visual", "Somatomotor", "DorsAttn", "SalVentAttn", "Limbic", "Control", "Default"]


def make_yeo_cereb_boxplot(df_sessions: pd.DataFrame) -> go.Figure:
    """
    Per-subject box plots: tSNR distribution across sessions for each Yeo network
    and whole-cerebellum GM. One subplot per subject, shared y-axis across subjects.

    Each box = distribution of session-level tSNR values for that region/subject.
    Cerebellum is shown alongside Yeo networks for direct comparison.
    """
    yeo_cols = [f"yeo_{n}" for n in YEO7_NAMES]
    cereb_col = "cereb_gm_mean" if "cereb_gm_mean" in df_sessions.columns else "cereb_mean"
    metric_cols = yeo_cols + [cereb_col]
    categories = YEO7_NAMES + ["Cerebellum"]

    subjects = sorted(df_sessions["subject"].unique())

    available_cols = [c for c in metric_cols if c in df_sessions.columns]
    if not available_cols:
        fig = go.Figure()
        fig.update_layout(title=_fig_title("Yeo–Cerebellum tSNR (no Yeo data)"), height=300)
        return fig

    # Global y range (shared across all subjects)
    all_vals = df_sessions[available_cols].values.ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    y_min = float(np.percentile(all_vals, 1)) * 0.95
    y_max = float(np.percentile(all_vals, 99)) * 1.05

    fig = make_subplots(rows=1, cols=len(subjects), subplot_titles=subjects, shared_yaxes=True)

    for col_idx, subj in enumerate(subjects, start=1):
        sdf = df_sessions[df_sessions["subject"] == subj]
        color = subject_color(subj)

        for col, label in zip(metric_cols, categories):
            if col not in sdf.columns:
                continue
            vals = sdf[col].dropna().values
            if len(vals) == 0:
                continue
            fig.add_trace(
                go.Box(
                    y=vals,
                    x=[label] * len(vals),
                    name=label,
                    marker_color=color,
                    line_color=color,
                    fillcolor=_hex_to_rgba(color, 0.4),
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    showlegend=False,
                    hovertemplate=f"{subj} {label}: %{{y:.1f}}<extra></extra>",
                ),
                row=1, col=col_idx,
            )

        fig.update_xaxes(tickangle=-45, row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text="tSNR", row=1, col=1)

    fig.update_yaxes(range=[y_min, y_max])
    fig.update_layout(
        title=_fig_title("Yeo Network + Cerebellum tSNR Distribution per Subject (session-level)"),
        height=480,
        showlegend=False,
        margin=dict(b=120),
    )
    return fig


def make_yeo_lobule_corr_matrix(
    df_sessions: pd.DataFrame,
    suit_label_names: List[str],
    min_sessions: int = 5,
) -> go.Figure:
    """
    Per-subject heatmap: Pearson r between each Yeo network and each SUIT lobule,
    computed across sessions.

    Rows = 7 Yeo networks, columns = SUIT lobules.
    Only plotted for subjects with >= min_sessions sessions.

    Note: correlations are across session-level tSNR means — they reflect
    shared scan-quality fluctuation, not functional connectivity.
    """
    subjects = sorted(df_sessions["subject"].unique())

    yeo_cols = [f"yeo_{n}" for n in YEO7_NAMES]
    suit_cols = [f"suit_{n}" for n in suit_label_names]

    # Filter to subjects with enough sessions and both Yeo + SUIT data
    eligible = []
    for subj in subjects:
        sdf = df_sessions[df_sessions["subject"] == subj]
        has_yeo = any(c in sdf.columns and sdf[c].notna().sum() >= min_sessions for c in yeo_cols)
        has_suit = any(c in sdf.columns and sdf[c].notna().sum() >= min_sessions for c in suit_cols)
        if has_yeo and has_suit and len(sdf) >= min_sessions:
            eligible.append(subj)

    if not eligible:
        fig = go.Figure()
        fig.update_layout(
            title=_fig_title(f"Yeo–Lobule tSNR Correlation (need ≥{min_sessions} sessions)"),
            height=200,
        )
        return fig

    fig = make_subplots(
        rows=1,
        cols=len(eligible),
        subplot_titles=eligible,
        shared_yaxes=True,
    )

    for col_idx, subj in enumerate(eligible, start=1):
        sdf = df_sessions[df_sessions["subject"] == subj]

        # Only keep columns with enough valid (non-NaN) sessions
        # This automatically excludes deep nuclei and other empty regions
        yeo_avail = [c for c in yeo_cols if c in sdf.columns and sdf[c].notna().sum() >= min_sessions]
        suit_avail = [c for c in suit_cols if c in sdf.columns and sdf[c].notna().sum() >= min_sessions]

        if not yeo_avail or not suit_avail:
            continue

        # Use complete rows across the retained columns
        combined = sdf[yeo_avail + suit_avail].dropna()
        if len(combined) < min_sessions:
            continue

        yeo_mat = combined[yeo_avail].values.astype(float)
        suit_mat = combined[suit_avail].values.astype(float)

        # Compute cross-correlation: combine → full corrcoef → take cross block
        combined = np.hstack([yeo_mat, suit_mat])       # (n_ses, n_yeo + n_lobules)
        full_corr = np.corrcoef(combined.T)              # (n_yeo + n_lobules, n_yeo + n_lobules)
        n_yeo = yeo_mat.shape[1]
        cross_corr = full_corr[:n_yeo, n_yeo:]          # (n_yeo, n_lobules)

        yeo_names_avail = [n for n, c in zip(YEO7_NAMES, yeo_cols) if c in sdf.columns]
        suit_names_avail = [n.replace("suit_", "") for n in suit_avail]

        fig.add_trace(
            go.Heatmap(
                z=cross_corr,
                x=suit_names_avail,
                y=yeo_names_avail,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                showscale=(col_idx == len(eligible)),
                colorbar=dict(title="Pearson r", x=1.02) if col_idx == len(eligible) else None,
                hovertemplate="Network: %{y}<br>Lobule: %{x}<br>r: %{z:.3f}<extra></extra>",
            ),
            row=1, col=col_idx,
        )
        if col_idx == 1:
            fig.update_yaxes(title_text="Yeo Network", row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=col_idx)

    fig.update_layout(
        title=_fig_title(
            "Yeo Network × SUIT Lobule tSNR Correlation (across sessions, per subject)\n"
            "⚠ Reflects shared scan-quality fluctuation, not functional connectivity"
        ),
        height=380,
        margin=dict(b=120, l=100),
    )
    return fig


def make_usable_volumes_bar(df_runs: pd.DataFrame) -> go.Figure:
    """Per-subject bar chart of % usable volumes per session (mean over runs)."""
    subjects = sorted(df_runs["subject"].unique())
    fig = go.Figure()

    for subj in subjects:
        subj_df = df_runs[df_runs["subject"] == subj].sort_values("session_num")
        sessions = subj_df.groupby("session")["pct_usable"].mean().reset_index()
        sessions["session_num"] = sessions["session"].apply(lambda s: int(s.split("-")[1]))
        sessions = sessions.sort_values("session_num")

        fig.add_trace(
            go.Bar(
                name=subj,
                x=sessions["session"].tolist(),
                y=sessions["pct_usable"].tolist(),
                marker_color=subject_color(subj),
                opacity=0.85,
                hovertemplate=f"{subj}<br>Session: %{{x}}<br>% Usable: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.add_hline(y=80, line_dash="dash", line_color="orange",
                  annotation_text="80% usable", annotation_position="right")
    fig.update_layout(
        title=_fig_title("% Usable Volumes After Scrubbing (FD>0.5 OR std_DVARS>1.5) per Session"),
        xaxis_title="Session",
        yaxis_title="% Usable volumes",
        yaxis_range=[0, 105],
        legend_title="Subject",
        barmode="group",
        height=450,
        xaxis_tickangle=-45,
    )
    return fig
