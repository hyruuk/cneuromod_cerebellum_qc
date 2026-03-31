#!/usr/bin/env python3
"""
Visualise the SUIT Anatom cerebellar atlas overlaid on the MNI152 template.

Produces a figure with three orthogonal views centred on the cerebellum,
with each of the 34 regions coloured according to the official TSV palette
and a legend mapping colours to region names.

Usage:
    python plot_atlas.py --atlas /path/to/suit_atl-Anatom_space-MNI_dseg.nii.gz
    python plot_atlas.py --atlas /path/to/atlas.nii.gz --output atlas_view.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from nilearn import datasets, plotting


def load_tsv_colors(tsv_path: Path) -> dict:
    """Return {index: (name, hex_color)} from the atlas TSV."""
    entries = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(row["index"])
            if idx == 0:
                continue
            entries[idx] = (row["name"], row["color"])
    return entries


def hex_to_rgba(hex_color: str, alpha: float = 0.85) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


def make_colormap(entries: dict) -> tuple[ListedColormap, BoundaryNorm, list]:
    """Build a discrete colormap aligned with atlas integer labels."""
    max_label = max(entries)
    # Slot 0 = background (transparent), slots 1..max_label = regions
    colors = [(0, 0, 0, 0)] * (max_label + 1)
    for idx, (name, hex_color) in entries.items():
        colors[idx] = hex_to_rgba(hex_color)
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, max_label + 1.5)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def main():
    parser = argparse.ArgumentParser(
        description="Plot SUIT cerebellar atlas overlaid on MNI152 template."
    )
    parser.add_argument(
        "--atlas",
        required=True,
        help="Path to suit_atl-Anatom_space-MNI_dseg.nii.gz",
    )
    parser.add_argument(
        "--output",
        default="atlas_view.png",
        help="Output image path (default: atlas_view.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150).",
    )
    args = parser.parse_args()

    atlas_path = Path(args.atlas)

    # Derive TSV path (same stem, .tsv extension)
    stem = atlas_path.name
    for suffix in (".gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    tsv_path = atlas_path.parent / (stem + ".tsv")

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Label TSV not found at {tsv_path}. "
            "Run download_suit_atlas.py first."
        )

    print(f"Loading atlas: {atlas_path}")
    print(f"Loading labels: {tsv_path}")

    entries = load_tsv_colors(tsv_path)
    cmap, norm = make_colormap(entries)

    atlas_img = nib.load(atlas_path)

    # MNI152 template as background (1mm, nilearn built-in)
    template_img = datasets.load_mni152_template(resolution=1)

    # Cutting coordinates centred on the cerebellum (MNI mm)
    # Axial slices: z = -45, -30, -15  (inferior → superior cerebellum)
    # Coronal slice: y = -60           (mid-cerebellum)
    # Sagittal slice: x = 0            (vermis)
    cut_coords_ax = (-45, -35, -25, -15)

    fig = plt.figure(figsize=(18, 12), facecolor="black")

    # --- Row 1: four axial slices ---
    ax_axial = fig.add_axes([0.0, 0.52, 0.72, 0.44])
    display = plotting.plot_anat(
        template_img,
        axes=ax_axial,
        display_mode="z",
        cut_coords=cut_coords_ax,
        draw_cross=False,
        annotate=True,
        black_bg=True,
        dim=-0.5,
    )
    display.add_overlay(
        atlas_img,
        cmap=cmap,
        vmin=0,
        vmax=len(entries),
        transparency=0.15,
    )
    ax_axial.set_title("Axial", color="white", fontsize=11, pad=4)

    # --- Row 2 left: coronal ---
    ax_cor = fig.add_axes([0.0, 0.05, 0.35, 0.44])
    display_cor = plotting.plot_anat(
        template_img,
        axes=ax_cor,
        display_mode="y",
        cut_coords=[-60],
        draw_cross=False,
        annotate=True,
        black_bg=True,
        dim=-0.5,
    )
    display_cor.add_overlay(
        atlas_img,
        cmap=cmap,
        vmin=0,
        vmax=len(entries),
        transparency=0.15,
    )
    ax_cor.set_title("Coronal  (y = −60 mm)", color="white", fontsize=11, pad=4)

    # --- Row 2 right: sagittal ---
    ax_sag = fig.add_axes([0.36, 0.05, 0.35, 0.44])
    display_sag = plotting.plot_anat(
        template_img,
        axes=ax_sag,
        display_mode="x",
        cut_coords=[0],
        draw_cross=False,
        annotate=True,
        black_bg=True,
        dim=-0.5,
    )
    display_sag.add_overlay(
        atlas_img,
        cmap=cmap,
        vmin=0,
        vmax=len(entries),
        transparency=0.15,
    )
    ax_sag.set_title("Sagittal  (x = 0 mm, vermis)", color="white", fontsize=11, pad=4)

    # --- Legend panel (right side) ---
    ax_leg = fig.add_axes([0.73, 0.02, 0.26, 0.95])
    ax_leg.set_axis_off()
    ax_leg.set_facecolor("black")

    patches = []
    for idx in sorted(entries):
        name, hex_color = entries[idx]
        patches.append(
            mpatches.Patch(color=hex_to_rgba(hex_color, alpha=1.0), label=f"{idx:2d}  {name}")
        )

    leg = ax_leg.legend(
        handles=patches,
        loc="center left",
        fontsize=7.5,
        frameon=False,
        labelcolor="white",
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0,
        labelspacing=0.4,
    )

    ax_leg.set_title(
        "SUIT Anatom regions",
        color="white",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=8,
    )

    fig.suptitle(
        "SUIT Cerebellar Atlas (Diedrichsen 2009) — MNI152 space",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="black")
    print(f"\n✓ Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
