# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Plots of a re-evaluation (needs matplotlib, `pip install calibconsumer[plot]`): the residuals as exaggerated arrows
and the reprojection error over the sensor. Written with matplotlib's object API – no display or pyplot state."""

from __future__ import annotations

import math

import numpy as np

ARROW_SPACING = 2.0  # the inliers' 90th percentile residual is drawn this many mean point spacings long,
ARROW_SHARE = (0.01, 0.05)  # within these shares of the image diagonal
OUTLIER_CAP = 3.0  # outlier arrows drawn at most this many times as long as the 90th percentile's
MAX_LEGEND = 10  # up to this many views get a legend with one colour each, more a colour scale


def _figure(resolution):
    from matplotlib.figure import Figure
    W, H = resolution
    width = 11.0
    fig = Figure(figsize=(width, width * H / W * 0.92 + 1.4))
    ax = fig.add_subplot()
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    ax.add_patch(_rectangle(W, H))
    return fig, ax


def _rectangle(W, H):
    from matplotlib.patches import Rectangle
    return Rectangle((-0.5, -0.5), W, H, fill=False, linewidth=0.8, edgecolor="0.55")


def _colours(count):
    import matplotlib
    if count <= MAX_LEGEND:
        palette = matplotlib.colormaps["tab10"]
        return [palette(k) for k in range(count)], None
    scale = matplotlib.colormaps["viridis"]
    return [scale(k / max(count - 1, 1)) for k in range(count)], scale


def residuals(evaluation, path, title=None, dpi=150):
    """Residual arrows (predicted − measured) from each measured point, exaggerated so that the inliers' 90th
    percentile is about two mean point spacings long; one colour per view; inliers as dots, outliers as crosses on top
    (their arrows capped at OUTLIER_CAP times that length)."""
    e = evaluation
    W, H = e.resolution
    fig, ax = _figure(e.resolution)
    use = e.valid & ~e.outlier
    p90 = float(np.percentile(e.reprojection[use], 90)) if use.any() else float("nan")
    diagonal = math.hypot(W, H)
    span = np.ptp(e.uv[e.valid], axis=0) if e.valid.any() else np.array([W, H])
    spacing = math.sqrt(max(float(span[0] * span[1]), 1.0) / max(int(e.valid.sum()), 1))
    length = float(np.clip(ARROW_SPACING * spacing, ARROW_SHARE[0] * diagonal, ARROW_SHARE[1] * diagonal))
    factor = length / p90 if p90 > 0 else 1.0
    capped = 0
    colours, scale = _colours(len(e.view_names))
    for k, name in enumerate(e.view_names):
        for outlier in (False, True):
            m = (e.view == k) & e.valid & (e.outlier == outlier)
            if not m.any():
                continue
            arrows = factor * e.residual[m]
            if outlier:
                drawn = np.linalg.norm(arrows, axis=-1, keepdims=True)
                cap = OUTLIER_CAP * length
                capped += int(np.sum(drawn > cap))
                arrows = np.where(drawn > cap, arrows * cap / np.maximum(drawn, 1e-300), arrows)
            ax.quiver(e.uv[m, 0], e.uv[m, 1], arrows[:, 0], arrows[:, 1], color=colours[k],
                      angles="xy", scale_units="xy", scale=1, width=0.0014 if not outlier else 0.0024,
                      headwidth=3.5, headlength=4, alpha=0.85 if not outlier else 1.0, zorder=2 if not outlier else 4)
            if outlier:
                ax.scatter(e.uv[m, 0], e.uv[m, 1], marker="X", s=70, linewidths=0.8, color=colours[k],
                           edgecolors="white", zorder=5)
            else:
                ax.scatter(e.uv[m, 0], e.uv[m, 1], marker="o", s=5, color=colours[k], zorder=3,
                           label=name if scale is None else None)
    invalid = ~e.valid
    if invalid.any():
        ax.scatter(e.uv[invalid, 0], e.uv[invalid, 1], marker="s", s=18, facecolors="none", edgecolors="0.3",
                   linewidths=0.8, zorder=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color="0.35", markersize=4, label="inlier"),
               Line2D([], [], marker="X", linestyle="", color="0.35", markeredgecolor="white", markersize=9,
                      label=f"outlier (Mahalanobis > {e.outlier_threshold:g})")]
    if invalid.any():
        handles.append(Line2D([], [], marker="s", linestyle="", markerfacecolor="none", color="0.3", markersize=5,
                              label="no prediction"))
    if scale is None:
        first = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="view", frameon=False)
        ax.add_artist(first)
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=8, frameon=False)
    else:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        bar = fig.colorbar(ScalarMappable(Normalize(0, len(e.view_names) - 1), scale), ax=ax, shrink=0.75, pad=0.02)
        bar.set_label("view (in name order)")
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)
    if math.isfinite(p90):
        ax.annotate("", xy=(0.03 * W + length, 0.97 * H), xytext=(0.03 * W, 0.97 * H),
                    arrowprops=dict(arrowstyle="->", color="0.15", lw=1.4))
        ax.text(0.03 * W, 0.955 * H, f"{p90:.3g} px", fontsize=8, color="0.15", va="bottom")
    s = e.summary()["all"]
    rms = (s["reprojectionPx"] or {}).get("rms")
    head = title + "\n" if title else ""
    cap_text = f", {capped} outlier arrows capped at ×{OUTLIER_CAP:g} the 90th percentile's length" if capped else ""
    ax.set_title(f"{head}Residuals predicted − measured, drawn ×{factor:.3g} (scale arrow: the inliers' 90th "
                 f"percentile, {p90:.3g} px)\n{s['valid']} points in {len(e.view_names)} views, {s['outliers']} "
                 f"outliers{cap_text}; inliers' RMS {'–' if rms is None else f'{rms:.3g}'} px", fontsize=10)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def error_map(evaluation, path, title=None, dpi=150):
    """The inliers' median reprojection error per sensor cell (cells of ⌈max(W/40, H/30)⌉ px), the outliers marked."""
    from matplotlib.colors import Normalize
    e = evaluation
    W, H = e.resolution
    step = int(math.ceil(max(W / 40, H / 30)))
    width, height = int(math.ceil(W / step)), int(math.ceil(H / step))
    use = e.valid & ~e.outlier
    cell = np.clip((e.uv[use] // step).astype(int), 0, [width - 1, height - 1])
    index = cell[:, 1] * width + cell[:, 0]
    values = np.full(width * height, np.nan)
    count = np.zeros(width * height, int)
    order = np.argsort(index, kind="stable")
    for key, group in _groups(index[order], e.reprojection[use][order]):
        values[key] = np.median(group)
        count[key] = len(group)
    values = values.reshape(height, width)
    fig, ax = _figure(e.resolution)
    finite = values[np.isfinite(values)]
    top = float(np.percentile(finite, 98)) if len(finite) else 1.0
    image = ax.imshow(np.ma.masked_invalid(values), cmap="magma", norm=Normalize(0, max(top, 1e-12)),
                      interpolation="nearest", extent=(-0.5, width * step - 0.5, height * step - 0.5, -0.5))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    outliers = e.valid & e.outlier
    if outliers.any():
        ax.scatter(e.uv[outliers, 0], e.uv[outliers, 1], marker="x", s=30, linewidths=1.2, color="#2a9df4",
                   label="outlier")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    bar = fig.colorbar(image, ax=ax, shrink=0.75, pad=0.02, extend="max" if len(finite) and finite.max() > top else
                       "neither")
    bar.set_label("median reprojection error [px]")
    s = e.summary()["all"]
    r = s["reprojectionPx"] or {}
    head = title + "\n" if title else ""
    ax.set_title(f"{head}Reprojection error over the sensor (inliers, median per {step} px cell; colours up to the "
                 f"98th percentile of the cells)\nRMS {r.get('rms', float('nan')):.3g} px, median "
                 f"{r.get('median', float('nan')):.3g} px, {s['outliers']} outliers", fontsize=10)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _groups(keys, values):
    if not len(keys):
        return
    edges = np.nonzero(np.diff(keys))[0] + 1
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(keys)]):
        yield int(keys[start]), values[start:stop]
