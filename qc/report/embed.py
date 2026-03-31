"""
Utilities for embedding static images (matplotlib) in HTML as base64 PNGs.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def fig_to_base64_png(fig: plt.Figure, dpi: int = 150) -> str:
    """
    Convert a matplotlib Figure to a base64-encoded PNG data URI.

    Returns
    -------
    String like 'data:image/png;base64,...'
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def tsnr_slices_to_html(
    tsnr_img: nib.Nifti1Image,
    subject: str,
    cut_coords: tuple = (-50, -40, -30, -20),
    vmax: Optional[float] = None,
    dpi: int = 130,
) -> str:
    """
    Render tSNR map as axial cerebellar slices using nilearn, return HTML <img> tag.

    Parameters
    ----------
    tsnr_img:
        3D NIfTI tSNR map.
    subject:
        Subject label (for title).
    cut_coords:
        Axial Z coordinates (mm MNI) to display — default covers cerebellum.
    vmax:
        Colorbar maximum. If None, uses 95th percentile of non-NaN values.
    dpi:
        Output resolution.
    """
    try:
        from nilearn import plotting as nlplot

        tsnr_data = tsnr_img.get_fdata()
        valid = tsnr_data[np.isfinite(tsnr_data)]
        if vmax is None and len(valid) > 0:
            vmax = float(np.percentile(valid, 95))
        if vmax is None or vmax <= 0:
            vmax = 100.0

        # Fetch MNI template for background
        try:
            from nilearn.datasets import load_mni152_template
            bg_img = load_mni152_template(resolution=2)
        except Exception:
            bg_img = None

        fig, ax = plt.subplots(1, 1, figsize=(12, 3), facecolor="black")
        display = nlplot.plot_stat_map(
            tsnr_img,
            bg_img=bg_img,
            display_mode="z",
            cut_coords=cut_coords,
            colorbar=True,
            title=f"{subject} — tSNR (cerebellar slices)",
            vmax=vmax,
            cmap="hot",
            axes=ax,
            black_bg=True,
        )
        data_uri = fig_to_base64_png(fig, dpi=dpi)
        return f'<img src="{data_uri}" style="max-width:100%;display:block;margin:auto;" />'
    except Exception as e:
        return f'<p style="color:orange;">[tSNR image unavailable: {e}]</p>'


def tsnr_interactive_viewer(
    tsnr_img: nib.Nifti1Image,
    subject: str,
    threshold: float = 5.0,
    vmax: Optional[float] = None,
    width: str = "100%",
    height: str = "500px",
) -> str:
    """
    Generate a nilearn interactive 3D viewer for the tSNR map.

    Returns an <iframe> tag with the self-contained viewer embedded via srcdoc.
    The viewer shows an interactive volume with clickable orthogonal views
    (axial/sagittal/coronal) + a 3D glass-brain perspective.

    Parameters
    ----------
    tsnr_img:
        3D NIfTI tSNR map.
    subject:
        Subject label for the figure title.
    threshold:
        Minimum tSNR value to display (lower values are transparent).
        Good default: 5.0 (suppresses background noise).
    width / height:
        CSS dimensions for the iframe.
    """
    try:
        from nilearn import plotting as nlplot

        tsnr_data = tsnr_img.get_fdata()
        valid = tsnr_data[np.isfinite(tsnr_data) & (tsnr_data > 0)]
        if vmax is None:
            vmax = float(np.percentile(valid, 97)) if len(valid) > 0 else 100.0

        # Replace NaN with 0 so nilearn does not emit "Non-finite values detected".
        # Zeros fall below threshold and are rendered transparent.
        clean_data = np.where(np.isfinite(tsnr_data), tsnr_data, 0.0).astype(np.float32)
        clean_img = nib.Nifti1Image(clean_data, tsnr_img.affine)

        try:
            from nilearn.datasets import load_mni152_template
            bg_img = load_mni152_template(resolution=2)
        except Exception:
            bg_img = None

        view = nlplot.view_img(
            clean_img,
            bg_img=bg_img,
            threshold=threshold,
            vmax=vmax,
            title=f"{subject} — tSNR (interactive)",
            colorbar=True,
            cmap="hot",
            symmetric_cmap=False,
        )
        # get_iframe() returns a self-contained <iframe> element
        return view.get_iframe(width=width, height=height)
    except Exception as e:
        return f'<p style="color:orange;">[Interactive viewer unavailable: {e}]</p>'


def atlas_interactive_viewer(
    atlas_img: nib.Nifti1Image,
    width: str = "100%",
    height: str = "500px",
) -> str:
    """
    Generate a nilearn interactive 3D viewer for the SUIT atlas.

    Returns an <iframe> tag with a self-contained viewer. Each region is
    assigned a distinct colour from a discrete colormap; the viewer allows
    click-to-navigate in orthogonal views.

    Parameters
    ----------
    atlas_img:
        3D integer NIfTI with SUIT label IDs (0 = background).
    width / height:
        CSS dimensions for the iframe.
    """
    try:
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        from nilearn import plotting as nlplot
        from nilearn.datasets import load_mni152_template

        atlas_data = np.asarray(atlas_img.dataobj, dtype=np.float32)
        n_labels = int(atlas_data.max())

        # Build a discrete colormap: one distinct colour per label.
        # tab20 + tab20b cover up to 40 colours.
        base_colors = (
            [plt.get_cmap("tab20")(i) for i in range(20)]
            + [plt.get_cmap("tab20b")(i) for i in range(20)]
        )
        # index 0 = transparent (background), indices 1..n_labels = region colours
        rgba = [(0, 0, 0, 0)] + base_colors[:n_labels]
        cmap = ListedColormap(rgba, name="suit_discrete")

        bg_img = load_mni152_template(resolution=2)

        # nilearn view_img requires finite values; background (0) renders as
        # transparent via threshold just below 1.
        view = nlplot.view_img(
            atlas_img,
            bg_img=bg_img,
            cmap=cmap,
            symmetric_cmap=False,
            vmin=0,
            vmax=n_labels,
            threshold=0.5,
            title="SUIT Cerebellar Atlas — click to navigate",
            colorbar=False,
        )
        return view.get_iframe(width=width, height=height)
    except Exception as e:
        return f'<p style="color:orange;">[Atlas viewer unavailable: {e}]</p>'


def array_to_base64_png(
    arr: np.ndarray,
    cmap: str = "RdBu_r",
    vmin: float = -3.0,
    vmax: float = 3.0,
    figsize: tuple = (12, 4),
    dpi: int = 100,
) -> str:
    """
    Render a 2D array as a heatmap image (for carpet plot fallback if Plotly is too large).
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    plt.colorbar(im, ax=ax, label="z-score")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    data_uri = fig_to_base64_png(fig, dpi=dpi)
    return f'<img src="{data_uri}" style="max-width:100%;display:block;margin:auto;" />'
