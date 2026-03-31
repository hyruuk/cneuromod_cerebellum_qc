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
    label_map: Optional[dict] = None,
    width: str = "100%",
    height: str = "500px",
) -> str:
    """
    Generate a nilearn interactive 3D viewer for the SUIT atlas, plus an
    HTML legend table showing region name, colour swatch, and MNI centroid.

    Returns an <iframe> followed by a colour-coded legend table.

    Parameters
    ----------
    atlas_img:
        3D integer NIfTI with SUIT label IDs (0 = background).
    label_map:
        Dict mapping integer label IDs → region name strings.
        If provided, a legend table is appended below the viewer.
    width / height:
        CSS dimensions for the iframe.
    """
    try:
        from matplotlib.colors import ListedColormap
        from nilearn import plotting as nlplot
        from nilearn.datasets import load_mni152_template

        atlas_data_int = np.asarray(atlas_img.dataobj, dtype=np.int32)
        atlas_data = atlas_data_int.astype(np.float32)
        n_labels = int(atlas_data.max())

        # Build discrete colormap: tab20 + tab20b cover up to 40 colours.
        # label_id k → base_colors[k-1] (index 0 = transparent background)
        base_colors = (
            [plt.get_cmap("tab20")(i) for i in range(20)]
            + [plt.get_cmap("tab20b")(i) for i in range(20)]
        )
        rgba = [(0, 0, 0, 0)] + base_colors[:n_labels]
        cmap = ListedColormap(rgba, name="suit_discrete")

        bg_img = load_mni152_template(resolution=2)

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
        iframe_html = view.get_iframe(width=width, height=height)

        # --- Legend table ---
        if label_map is None:
            return iframe_html

        affine = atlas_img.affine

        def _rgba_to_css(r, g, b, a) -> str:
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        rows_html = ""
        for label_id in sorted(label_map.keys()):
            name = label_map[label_id]
            color_rgba = base_colors[label_id - 1] if label_id - 1 < len(base_colors) else (0.5, 0.5, 0.5, 1.0)
            css_color = _rgba_to_css(*color_rgba)

            # Centroid in MNI mm
            voxels = np.argwhere(atlas_data_int == label_id)
            if len(voxels) > 0:
                centroid_vox = voxels.mean(axis=0)
                centroid_mni = affine @ np.append(centroid_vox, 1.0)
                cx, cy, cz = centroid_mni[:3]
                coord_str = f"({cx:.0f}, {cy:.0f}, {cz:.0f})"
            else:
                coord_str = "—"

            rows_html += (
                f"<tr>"
                f'<td style="text-align:center;">{label_id}</td>'
                f'<td><span style="display:inline-block;width:18px;height:18px;'
                f'background:{css_color};border:1px solid #999;border-radius:3px;'
                f'vertical-align:middle;"></span></td>'
                f"<td>{name}</td>"
                f'<td style="font-family:monospace;color:#555;">{coord_str}</td>'
                f"</tr>\n"
            )

        legend_html = f"""
<details style="margin-top:12px;">
  <summary style="cursor:pointer;font-weight:bold;color:#636EFA;">
    Atlas Legend — {n_labels} regions (click to expand)
  </summary>
  <div style="max-height:400px;overflow-y:auto;margin-top:8px;">
    <table style="border-collapse:collapse;width:100%;font-size:0.88em;">
      <thead>
        <tr style="background:#636EFA;color:white;">
          <th style="padding:6px 10px;">ID</th>
          <th style="padding:6px 10px;">Colour</th>
          <th style="padding:6px 10px;text-align:left;">Region</th>
          <th style="padding:6px 10px;text-align:left;">MNI centroid (x,y,z mm)</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</details>"""

        return iframe_html + legend_html

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
