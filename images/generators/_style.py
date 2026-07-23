"""Shared visual style for every figure generator under images/generators/.

A restrained, journal-figure palette (near-black ink on white, two or three
sparse accent colors) replacing per-script ad hoc palettes so the whole
figure set reads as one system, modeled on the CatPred-DB / CatPred figure
style (Boorla & Frazier et al.): thin unfilled outlines, minimal color, bold
lettered panel labels.

Run: not a script; imported by the other generators in this directory.
"""
import logging
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

OUT_DIR = Path(__file__).resolve().parent.parent / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Palette
# ═══════════════════════════════════════════════════════════════════════════
INK = "#111111"
INK_SECONDARY = "#454545"
INK_MUTED = "#8a8a8a"
SURFACE = "#ffffff"

#  categorical accents (CVD-checked as a set via dataviz:validate_palette.js;
#  colors never carry identity alone - every swatch above pairs with a text
#  label, satisfying the tool's secondary-encoding exception)
NAVY = "#2d5f8a"     # primary accent: protein modality / data provenance
MAGENTA = "#b5396f"  # secondary accent: contrast category (mutant, reserved/held-out)
TEAL = "#1f8f7f"     # tertiary accent: 3D / geometric branch
AMBER = "#c17f16"    # physics-derived / physically-constrained quantity
GOOD = "#2f8f45"     # final validated artifact / output, used sparingly

FONT_STACK = ["Helvetica", "Arial", "DejaVu Sans"]

# ═══════════════════════════════════════════════════════════════════════════
#  Shared text-to-edge margins (one constant per edge, used by every box-like
#  helper below, so padding reads as one consistent system rather than each
#  helper inventing its own close-but-different inset).
# ═══════════════════════════════════════════════════════════════════════════
PAD_TOP = 0.22
PAD_SIDE = 0.22
PAD_BOTTOM = 0.14
TITLE_SUBTITLE_GAP = 0.30


def new_figure(fig_w: float, fig_h: float):
    """A blank figure/axes pair in data coordinates == inches, white background.

    Parameters
    ----------
    fig_w : float
        Figure width, in inches (== data-coordinate width, since the axes
        span exactly (0, fig_w) x (0, fig_h)).
    fig_h : float
        Figure height, in inches (== data-coordinate height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure, white-background, at 200 dpi.
    ax : matplotlib.axes.Axes
        A single axes spanning the whole figure with equal aspect, no
        ticks/spines, in data coordinates equal to inches.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_STACK
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def panel_label(ax, x: float, y: float, letter: str) -> None:
    """Bold lower-case panel letter (a, b, c...), top-left of a panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on, in the data-coordinates-as-inches system set up
        by `new_figure`.
    x, y : float
        Top-left anchor of the letter, in data coordinates.
    letter : str
        The panel letter to draw (e.g. "a", "b").

    Returns
    -------
    None
    """
    ax.text(x, y, letter, ha="left", va="top", fontsize=16, fontweight="bold", color=INK, zorder=5)


def box(ax, x, y, w, h, color=INK, title="", subtitle="", caption=None, emphasis=False,
        dashed=False, title_size=11.0, sub_size=8.6, wrap=26, fill_alpha=0.0, linespacing=1.42):
    """A thin-outline, near-unfilled rounded box with a bold title and muted subtitle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : float
        Lower-left corner of the box, in data coordinates.
    w, h : float
        Box width and height, in data coordinates.
    color : str, optional
        Outline (and, if `fill_alpha` > 0, fill) color. Defaults to `INK`.
    title : str, optional
        Bold title text, top-centered; supports an embedded "\\n" for a
        multi-line title.
    subtitle : str, optional
        Muted secondary text below the title, word-wrapped to `wrap`
        characters unless it already contains a "\\n" (treated as
        pre-broken, e.g. one LaTeX span per line, and left unwrapped).
    caption : str, optional
        Small italic monospace caption, bottom-centered (e.g. a table/file
        name backing the box).
    emphasis : bool, optional
        Draw a thicker outline to highlight a terminal/output box.
    dashed : bool, optional
        Draw a dashed outline instead of solid, for a reserved/pending box.
    title_size, sub_size : float, optional
        Font sizes for `title` and `subtitle`.
    wrap : int, optional
        Character width `subtitle` is wrapped to (ignored if `subtitle`
        already contains "\\n").
    fill_alpha : float, optional
        Fill opacity for `color`; 0 (default) renders an unfilled outline.
    linespacing : float, optional
        Line spacing for a wrapped/multi-line `subtitle`.

    Returns
    -------
    dict
        Geometry anchors for connecting this box to others: `x`, `y`, `w`,
        `h`, `cx`, `cy` (center), `l`, `r`, `t`, `b` (left/right/top/bottom
        edges) - the shape consumed by `arrow`, `route`, and `leader`.
    """
    if fill_alpha > 0:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=0, facecolor=color, alpha=fill_alpha, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=1.9 if emphasis else 1.2, edgecolor=color, facecolor="none",
        linestyle=(0, (5, 3)) if dashed else "-", zorder=3))
    cx = x + w / 2
    ty = y + h - PAD_TOP
    if title:
        ax.text(cx, ty, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=INK, zorder=4)
    if subtitle:
        # A multi-line title (embedded "\n") pushes its own bottom edge down
        # by one line per extra line - the gap below it must grow to match,
        # or the subtitle collides with the title's later lines.
        extra_title_lines = title.count("\n") if title else 0
        sy = ty - TITLE_SUBTITLE_GAP - extra_title_lines * (title_size / 72 * 1.25) if title else y + h - PAD_TOP
        # Pre-broken text (explicit "\n", e.g. one LaTeX $...$ span per line)
        # must not be re-wrapped: textwrap.fill would split a math span
        # across lines and matplotlib would render it as literal source.
        wrapped = subtitle if "\n" in subtitle else textwrap.fill(subtitle, width=wrap)
        ax.text(cx, sy, wrapped, ha="center", va="top",
                fontsize=sub_size, color=INK_SECONDARY, linespacing=linespacing, zorder=4)
    if caption:
        ax.text(cx, y + PAD_BOTTOM, caption, ha="center", va="bottom", fontsize=7.6,
                style="italic", color=INK_MUTED, family="monospace", zorder=4)
    return dict(x=x, y=y, w=w, h=h, cx=cx, cy=y + h / 2, l=x, r=x + w, t=y + h, b=y)


def container(ax, x, y, w, h, color, title, caption=None):
    """A dashed containing frame around a group of boxes (no fill).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : float
        Lower-left corner of the frame, in data coordinates.
    w, h : float
        Frame width and height, in data coordinates.
    color : str
        Outline, title, and caption color.
    title : str
        Bold title text, top-centered.
    caption : str, optional
        Small italic monospace caption, bottom-centered.

    Returns
    -------
    dict
        Geometry anchors (`x`, `y`, `w`, `h`, `cx`, `cy`, `l`, `r`, `t`,
        `b`), matching `box`'s return shape for use with `arrow`/`route`.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10",
        linewidth=1.3, edgecolor=color, facecolor="none",
        linestyle=(0, (6, 3)), zorder=1))
    ax.text(x + w / 2, y + h - PAD_TOP, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK, zorder=4)
    if caption:
        ax.text(x + w / 2, y + PAD_BOTTOM, caption, ha="center", va="bottom", fontsize=7.6,
                style="italic", color=INK_MUTED, family="monospace", zorder=4)
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, l=x, r=x + w, t=y + h, b=y)


_SIDE_POINT = {
    "right": lambda b: (b["x"] + b["w"], b["cy"]),
    "left": lambda b: (b["x"], b["cy"]),
    "bottom": lambda b: (b["cx"], b["y"]),
    "top": lambda b: (b["cx"], b["y"] + b["h"]),
}


def arrow(ax, b_from, b_to, from_side="right", to_side="left", color=INK_SECONDARY, dashed=False, lw=1.3):
    """A straight arrow between two box/container geometry dicts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    b_from, b_to : dict
        Geometry dicts as returned by `box`/`container`/`field_table`.
    from_side, to_side : str, optional
        Which edge of `b_from`/`b_to` the arrow starts/ends at - one of
        "right", "left", "bottom", "top".
    color : str, optional
        Arrow color.
    dashed : bool, optional
        Draw a dashed shaft instead of solid.
    lw : float, optional
        Line width.

    Returns
    -------
    None
    """
    p0 = _SIDE_POINT[from_side](b_from)
    p1 = _SIDE_POINT[to_side](b_to)
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=10, linewidth=lw, color=color,
        linestyle=(0, (4, 2.5)) if dashed else "-", shrinkA=0, shrinkB=0, zorder=1))


def route(ax, pts, color=INK_SECONDARY, dashed=False, lw=1.3):
    """Orthogonal (Manhattan) connector through `pts`, arrowhead on the final segment.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    pts : list of tuple of float
        Ordered (x, y) waypoints, in data coordinates, the connector passes
        through; the arrowhead is drawn on the final segment (`pts[-2]` to
        `pts[-1]`).
    color : str, optional
        Line and arrowhead color.
    dashed : bool, optional
        Draw a dashed line instead of solid.
    lw : float, optional
        Line width.

    Returns
    -------
    None
    """
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
    ax.add_patch(PathPatch(
        MPath(pts, codes), fill=False, edgecolor=color, linewidth=lw,
        linestyle=(0, (4, 2.5)) if dashed else "-", joinstyle="round", capstyle="round", zorder=1))
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0, zorder=1))


MONO = {"family": "monospace"}


def field_table(ax, x, y, w, h, color, title, tag, rows, row_fs=8.0, dashed=False):
    """A titled panel listing (name, shape, dtype) rows in a monospace grid,
    used for tensor/schema reference figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : float
        Lower-left corner of the panel, in data coordinates.
    w, h : float
        Panel width and height, in data coordinates.
    color : str
        Outline, tag, and rule-line color.
    title : str
        Bold title text, top-left.
    tag : str
        Small italic tag text, top-right (e.g. a shape/count summary).
    rows : list of tuple of str
        (name, shape, dtype) triples, rendered one per line in a fixed-
        width monospace grid, top to bottom.
    row_fs : float, optional
        Row font size. Row pitch shrinks toward a legible floor so any
        number of rows fits within `h`.
    dashed : bool, optional
        Draw a dashed outline instead of solid.

    Returns
    -------
    dict
        Geometry anchors (`x`, `y`, `w`, `h`, `cx`, `cy`, `l`, `r`, `t`,
        `b`), matching `box`'s return shape for use with `arrow`/`leader`.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=1.3, edgecolor=color, facecolor="none",
        linestyle=(0, (5, 3)) if dashed else "-", zorder=3))
    ax.text(x + PAD_SIDE, y + h - PAD_TOP, title, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color=INK, zorder=4)
    ax.text(x + w - PAD_SIDE, y + h - PAD_TOP, tag, ha="right", va="top", fontsize=7.8,
            style="italic", color=color, zorder=4)
    ax.plot([x + PAD_SIDE - 0.04, x + w - PAD_SIDE + 0.04], [y + h - PAD_TOP - 0.28] * 2,
            color=color, linewidth=0.8, alpha=0.5, zorder=4)
    # Row pitch shrinks (down to a legible floor) so any row count fits
    # within the given panel height instead of overflowing its border.
    row_step = min(0.335, max(0.235, (h - 0.68) / max(1, len(rows))))
    ry = y + h - PAD_TOP - 0.48
    for name, shape, dtype in rows:
        line = f"{name:<20}{shape:<10}{dtype}"
        ax.text(x + PAD_SIDE + 0.02, ry, line, ha="left", va="top", fontsize=row_fs,
                color=INK_SECONDARY, zorder=4, **MONO)
        ry -= row_step
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, l=x, r=x + w, t=y + h, b=y)


def leader(ax, p_from, p_to, color=INK_MUTED):
    """A muted dashed leader line from one box's right edge to another's left edge.

    Used to connect a panel to the detail table that expands one of its
    fields (e.g. an atom-features column linked out from a tensor-shape
    table row).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p_from, p_to : dict
        Geometry dicts as returned by `box`/`field_table`; the leader runs
        from `p_from`'s right edge (`r`, `cy`) to `p_to`'s left edge (`l`,
        `cy`).
    color : str, optional
        Line and arrowhead color.

    Returns
    -------
    None
    """
    ax.add_patch(FancyArrowPatch(
        (p_from["r"], p_from["cy"]), (p_to["l"], p_to["cy"]),
        arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=color,
        linestyle=(0, (4, 2)), shrinkA=2, shrinkB=2, zorder=1))


def legend(ax, x: float, y: float, items, gap: float = 2.4, dashed_flags=None):
    """A row of small color swatches with labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : float
        Lower-left anchor of the first swatch, in data coordinates.
    items : list of tuple of (str, str)
        (color, label) pairs, drawn left to right.
    gap : float, optional
        Horizontal spacing between successive swatches, in data
        coordinates.
    dashed_flags : list of bool, optional
        Per-item flag drawing that swatch's outline dashed instead of
        solid; defaults to all solid if omitted.

    Returns
    -------
    None
    """
    for i, (color, label) in enumerate(items):
        dashed = bool(dashed_flags and dashed_flags[i])
        sx = x + i * gap
        ax.add_patch(Rectangle((sx, y), 0.24, 0.24, facecolor=color, alpha=0.85,
                               edgecolor=color, linewidth=1.1,
                               linestyle="--" if dashed else "-", zorder=4))
        ax.text(sx + 0.36, y + 0.12, label, ha="left", va="center", fontsize=8.4,
                color=INK_SECONDARY, zorder=4)


def footnote(ax, fig_w: float, text: str, y: float = 0.20) -> None:
    """A small italic footnote, right-aligned near the bottom of the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    fig_w : float
        Figure width, in data coordinates; the footnote is right-anchored
        near `fig_w`.
    text : str
        Footnote text.
    y : float, optional
        Vertical anchor, in data coordinates from the bottom of the figure.

    Returns
    -------
    None
    """
    ax.text(fig_w - 0.35, y, text, ha="right", va="bottom", fontsize=7.3,
            color=INK_MUTED, style="italic")


def save(fig, name: str) -> Path:
    """Write `fig` to images/generated/{name}.svg, plus an optional PNG preview.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Output file stem (no extension).

    Returns
    -------
    Path
        Path to the written SVG. If the `THERMOKP_FIG_PREVIEW_DIR`
        environment variable is set, a PNG preview is also written there,
        for a faster feedback loop than opening the SVG on every change.
    """
    svg_path = OUT_DIR / f"{name}.svg"
    fig.savefig(svg_path, facecolor=SURFACE)
    logger.info(f"wrote {svg_path}")
    preview_dir = os.environ.get("THERMOKP_FIG_PREVIEW_DIR")
    if preview_dir:
        png_path = Path(preview_dir) / f"{name}.png"
        fig.savefig(png_path, facecolor=SURFACE, dpi=150)
        logger.info(f"wrote preview {png_path}")
    return svg_path
