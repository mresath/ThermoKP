"""Generates three model-results figures, all built entirely from
src/evaluation/evaluate_dataset.py's merged results store - this script
performs no inference itself and never queries
thermokp_database.db directly, so every figure always reflects an actual
evaluated checkpoint:

1. results_summary_pinn.svg - kcat/Km parity plots and a percentile-of-error
   breakdown for 'val' and 'benchmark' (PINN).
2. results_crossmodel_comparison.svg - cross-model comparison for 'benchmark' on <=40% seq ID.
3. results_r2_comparison.svg - kcat/Km R^2, wild-type vs. mutant (grouped),
   validation vs. benchmark (colored).
4. results_error_histograms.svg - kcat/Km absolute log-error density,
   wild-type vs. mutant, one panel per (parameter, split) combination.

Run: python -m images.generators.results_summary [--eval_path data/results/eval.json]
    [--model_type pinn]
Output: images/generated/results_summary_pinn.svg,
    images/generated/results_summary_baseline_vs_pinn.svg,
    results_r2_comparison.svg, results_error_histograms.svg

Known Caveats:
- Requires data/results/eval.json (the single shared store every
  evaluate_dataset.py run merges into, keyed by "{split}_{model_type}"),
  written by `uv run python -m src.evaluation.evaluate_dataset [--split ...]
  [--model_type ...]`. Every figure needs at least one of 'val'/'benchmark'
  present for the requested model_type; figures 2 and 3 specifically need
  both (their whole point is comparing the two), so those two raise
  FileNotFoundError naming the exact command for whichever run_key is
  missing if only one split has been evaluated so far. Figures 1 and 2 instead
  render whichever split(s) are available and log a note for any that aren't.
- The cross-model comparison row in figure 2 requires seq_id_<=40 data to be present.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import AMBER, INK, INK_MUTED, INK_SECONDARY, MAGENTA, NAVY, SURFACE, save

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH_DEFAULT = "data/results/eval.json"

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

PERCENTILE_BIN_ORDER = ["p0-25", "p25-50", "p50-75", "p75-90", "p90-95", "p95-99", "p99-100"]
VALIDATION_COLOR, BENCHMARK_COLOR = NAVY, AMBER
SPLIT_LABELS = {"val": "Validation", "benchmark": "Benchmark"}
MODEL_DISPLAY_NAMES = {
    "pinn": "ThermoKP PINN",
    "baseline": "ThermoKP Baseline",
    "dlkcat": "DLKCat",
    "unikp": "UniKP",
    "catpred": "CatPred",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Shared data loading
# ═══════════════════════════════════════════════════════════════════════════
def _load_eval_store(eval_path: Path) -> dict:
    """Load the shared evaluation results store written by evaluate_dataset.py.

    Parameters
    ----------
    eval_path : Path
        Path to the JSON results store.

    Returns
    -------
    dict
        The store, keyed by "{split}_{model_type}".

    Raises
    ------
    FileNotFoundError
        If `eval_path` does not exist, naming the evaluate_dataset.py
        command to run first.
    """
    if not eval_path.exists():
        raise FileNotFoundError(
            f"No evaluation results store at {eval_path}. Run this first: "
            f"uv run python -m src.evaluation.evaluate_dataset"
        )
    return json.loads(eval_path.read_text(encoding="utf-8"))


def _get_run(store: dict, split: str, model_type: str) -> dict:
    """Look up one evaluation run in the results store.

    Parameters
    ----------
    store : dict
        Results store, as returned by `_load_eval_store`.
    split : str
        "val" or "benchmark".
    model_type : str
        One of "pinn", "baseline", "dlkcat", "unikp", "catpred".

    Returns
    -------
    dict
        The run's evaluation record.

    Raises
    ------
    FileNotFoundError
        If no run for `split`/`model_type` exists in `store`, naming the
        evaluate_dataset.py command to run first.
    """
    run_key = f"{split}_{model_type}"
    if run_key not in store:
        raise FileNotFoundError(
            f"No evaluation run for split='{split}', model_type='{model_type}' "
            f"(key '{run_key}') in the results store. Run this first: "
            f"uv run python -m src.evaluation.evaluate_dataset --split {split} --model_type {model_type}"
        )
    return store[run_key]


def _style_axes(ax) -> None:
    """Apply the shared journal-figure axes style: white background, no
    top/right spines, muted left/bottom spines and tick labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to style, in place.

    Returns
    -------
    None
    """
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)


def _new_fig(fig_w: float, fig_h: float):
    """Create a blank white-background figure at the shared journal-figure font/dpi.

    Parameters
    ----------
    fig_w, fig_h : float
        Figure width and height, in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure, with axes/subplots left for the caller to add.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1: parity plots, error concentration, PINN-vs-baseline
# ═══════════════════════════════════════════════════════════════════════════
def _parity_range(predictions_list: list, log_pred_col: str, log_true_col: str) -> tuple:
    """Shared axis range across every DataFrame in `predictions_list`, so
    parity panels for the same parameter (e.g. kcat Validation vs. kcat
    Benchmark) always plot on identical scales - mismatched per-panel
    ranges would make one split's spread look better/worse than it is."""
    lo = min(min(p[log_true_col].min(), p[log_pred_col].min()) for p in predictions_list) - 0.3
    hi = max(max(p[log_true_col].max(), p[log_pred_col].max()) for p in predictions_list) + 0.3
    return lo, hi


def _parity_panel(ax, predictions: pd.DataFrame, overall: dict, log_pred_col: str, log_true_col: str,
                  symbol: str, axis_range: tuple) -> None:
    """Draw one true-vs-predicted parity scatter panel, wild-type vs. mutant.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    predictions : pandas.DataFrame
        Per-record predictions, with a `mutation` column and the
        `log_pred_col`/`log_true_col` value columns.
    overall : dict
        Aggregate metrics for this parameter/split: "r2", "rmse", "p1mag",
        "n", annotated in the panel's stats box.
    log_pred_col, log_true_col : str
        Column names in `predictions` holding the log10 predicted/true
        values.
    symbol : str
        Axis-label symbol for the parameter (e.g. "$\\log_{10}k_{cat}$").
    axis_range : tuple of float
        Shared (lo, hi) axis range, from `_parity_range`, so this panel
        plots on the same scale as its sibling split/parameter panels.

    Returns
    -------
    None
    """
    is_mutant = predictions["mutation"].fillna("").astype(str).str.len() > 0
    for mask, color, label in [(~is_mutant, NAVY, "Wild-Type"), (is_mutant, MAGENTA, "Mutant")]:
        ax.scatter(predictions.loc[mask, log_true_col], predictions.loc[mask, log_pred_col],
                  s=10, alpha=0.5, color=color, edgecolor="none", label=label)
    lo, hi = axis_range
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"true {symbol}", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(f"predicted {symbol}", fontsize=9, color=INK_SECONDARY)
    stats_text = (f"$R^2$ = {overall['r2']:.3f}\n"
                 f"RMSE = {overall['rmse']:.3f}\n"
                 f"p1mag = {overall['p1mag']:.1f}%\n"
                 f"n = {overall['n']:,}")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, ha="left", va="top",
           fontsize=8.5, color=INK, linespacing=1.5)
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def _percentile_panel(ax, kcat_bins: dict, km_bins: dict, y_max: float) -> None:
    """Draw a grouped-bar panel of per-percentile-bin RMSE for kcat and Km.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    kcat_bins, km_bins : dict
        {percentile bin label: {"rmse": float, ...}} for kcat and Km
        respectively, keyed by the bins present in `PERCENTILE_BIN_ORDER`.
    y_max : float
        Shared y-axis ceiling, so this panel plots on the same scale as
        its sibling split panels.

    Returns
    -------
    None
    """
    labels = [b for b in PERCENTILE_BIN_ORDER if b in kcat_bins]
    x = np.arange(len(labels))
    width = 0.36
    kcat_rmse = [kcat_bins[b]["rmse"] for b in labels]
    km_rmse = [km_bins[b]["rmse"] for b in labels]
    ax.bar(x - width / 2, kcat_rmse, width, color=NAVY, label=r"$k_{cat}$")
    ax.bar(x + width / 2, km_rmse, width, color=MAGENTA, label=r"$K_m$")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("percentile of |log-error|", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("RMSE within bin", fontsize=9, color=INK_SECONDARY)
    ax.legend(loc="upper left", fontsize=9, frameon=False)


def generate_crossmodel_comparison(store: dict) -> None:
    """Generate the cross-model RMSE/R^2/p1mag comparison figure on the
    benchmark split's <=40% sequence-identity subset.

    Renders one row per parameter (kcat, Km) and one column per metric
    (RMSE, R^2, p1mag), one bar per model with seq_id_<=40 data available;
    models without that subset are skipped and logged. Saves
    "results_crossmodel_comparison" via `_style.save` (no-op, only logging
    a note, if no model has the required data).

    Parameters
    ----------
    store : dict
        Results store, as returned by `_load_eval_store`.

    Returns
    -------
    None
    """
    models = ["pinn", "baseline", "dlkcat", "unikp", "catpred"]

    runs = {}
    for m in models:
        try:
            run = _get_run(store, "benchmark", m)
            if "seq_id_union_cutoffs" in run and "seq_id_<=40" in run["seq_id_union_cutoffs"]:
                runs[m] = run["seq_id_union_cutoffs"]["seq_id_<=40"]
            else:
                logger.info(f"No seq_id_union_<=40 data for {m}, skipping in crossmodel comparison.")
        except FileNotFoundError:
            pass

    if not runs:
        logger.info("No models with union seq_id_<=40 data available for crossmodel comparison.")
        return

    fig = _new_fig(13.333, 5.6 * 2)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                          left=0.055, right=0.97, top=1 - 0.16 / 2, bottom=0.12)

    letters = iter("abcdef")
    metrics = [("rmse", "RMSE (Log-Space)"), ("r2", r"$R^2$"), ("p1mag", "p1mag (%)")]
    model_labels = [m for m in models if m in runs]

    for row, (param, symbol, color) in enumerate([("kcat", r"$k_{cat}$", NAVY), ("km", r"$K_m$", MAGENTA)]):
        row_models = [m for m in model_labels if not (param == "km" and m == "dlkcat")]

        for col, (key, m_label) in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])
            _style_axes(ax)

            vals = []
            for m in row_models:
                overall = runs[m][param].get("overall", {})
                vals.append(overall.get(key, float("nan")))

            x = np.arange(len(row_models))
            bars = ax.bar(x, vals, width=0.42, color=color)
            display_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in row_models]
            ax.set_xticks(x, display_labels, fontsize=7.5, rotation=30, ha="center")

            title = f"{symbol} {m_label}"
            ax.set_title(title, fontsize=10, color=INK_SECONDARY, loc="center")

            valid_vals = [v for v in vals if not np.isnan(v)]
            headroom = max(abs(v) for v in valid_vals) * 0.18 or 0.1 if valid_vals else 0.1
            if valid_vals:
                ax.set_ylim(min(0, min(valid_vals) - headroom), max(valid_vals) + headroom)

            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                va = "bottom" if v >= 0 else "top"
                format_str = f"{v:.3f}" if key != "p1mag" else f"{v:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, v + (headroom * 0.15 if v >= 0 else -headroom * 0.15),
                        format_str, ha="center", va=va, fontsize=7.6, color=INK)

            pos = gs[row, col].get_position(fig)
            fig.text(pos.x0 - 0.035, pos.y1 + 0.42 / fig.get_figheight(), next(letters),
                     fontsize=15, fontweight="bold", color=INK, ha="left", va="bottom")



    # Single footer at the very bottom of the figure
    shared_n_kcat_final = next(
        (int(runs[m]["kcat"]["overall"]["n"]) for m in models if m in runs), 0
    )
    n_note = f"n={shared_n_kcat_final}" if shared_n_kcat_final > 0 else "n unavailable"
    fig.text(
        0.5, 0.42 / (5.6 * 2) / 2,
        f"Benchmark entries with training-set sequence identity ≤ 40% (union of all model training sets) — {n_note}",
        ha="center", va="center", fontsize=7.5, color=INK_MUTED,
        transform=fig.transFigure,
    )

    save(fig, "results_crossmodel_comparison")


def generate_results_summary(store: dict, model_type: str) -> None:
    """Generate the parity and error-concentration figure for `model_type`.

    Renders one row per available split ('val', 'benchmark'), each with a
    kcat parity panel, a Km parity panel, and an error-concentration
    (percentile-of-error) panel, all sharing axis ranges across splits.
    Saves "results_summary_pinn" via `_style.save`.

    Parameters
    ----------
    store : dict
        Results store, as returned by `_load_eval_store`.
    model_type : str
        Model type to render (e.g. "pinn").

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If `store` has no run for `model_type` on either 'val' or
        'benchmark'.
    """
    pinn_specs: list = []
    try:
        pinn_val = _get_run(store, "val", model_type)
        pinn_specs.append(("parity", "val", pinn_val))
    except FileNotFoundError as e:
        logger.info(f"No {model_type} evaluation for split='val' - skipping PINN validation row: {e}")

    try:
        pinn_bench = _get_run(store, "benchmark", model_type)
        pinn_specs.append(("parity", "benchmark", pinn_bench))
    except FileNotFoundError as e:
        logger.info(f"No {model_type} evaluation for split='benchmark' - skipping benchmark rows: {e}")

    all_parity_specs = [spec for spec in pinn_specs if spec[0] == "parity"]
    if not all_parity_specs:
        raise FileNotFoundError(
            f"No '{model_type}' evaluation for either 'val' or 'benchmark' in the results store. "
            f"Run this first: uv run python -m src.evaluation.evaluate_dataset --model_type {model_type}"
        )

    # Shared axis ranges across every parity row so Validation and Benchmark
    # panels for the same parameter are never plotted on deceptively
    # different scales.
    parity_predictions = [pd.DataFrame(spec[2]["records"]) for spec in all_parity_specs]
    kcat_range = _parity_range(parity_predictions, "log_kcat_pred", "log_kcat_true")
    km_range = _parity_range(parity_predictions, "log_km_pred", "log_km_true")
    
    pct_y_max = 0.0
    for spec in all_parity_specs:
        if spec[2]["kcat"]["percentile_bins"]:
            pct_y_max = max(pct_y_max, max(b["rmse"] for b in spec[2]["kcat"]["percentile_bins"].values()))
        if spec[2]["km"]["percentile_bins"]:
            pct_y_max = max(pct_y_max, max(b["rmse"] for b in spec[2]["km"]["percentile_bins"].values()))
    pct_y_max *= 1.08

    def _draw_figure(specs: list, filename: str):
        n_rows = len(specs)
        fig = _new_fig(13.333, 5.6 * n_rows)
        gs = fig.add_gridspec(n_rows, 3, hspace=0.55, wspace=0.38,
                             left=0.055, right=0.97, top=1 - 0.16 / n_rows, bottom=0.42 / (5.6 * n_rows))

        letters = iter("abcdefghijklmnopqrstuvwxyz")
        for row, spec in enumerate(specs):
            if spec[0] == "parity":
                _, split, pinn_run = spec
                split_label = SPLIT_LABELS[split]
                ax_kcat = fig.add_subplot(gs[row, 0])
                ax_km = fig.add_subplot(gs[row, 1])
                ax_pct = fig.add_subplot(gs[row, 2])
                for ax in (ax_kcat, ax_km, ax_pct):
                    _style_axes(ax)
                for col in range(3):
                    pos = gs[row, col].get_position(fig)
                    fig.text(pos.x0 - 0.035, pos.y1 + 0.42 / fig.get_figheight(), next(letters),
                            fontsize=15, fontweight="bold", color=INK, ha="left", va="bottom")

                pinn_predictions = pd.DataFrame(pinn_run["records"])
                ax_kcat.set_title(rf"$k_{{cat}}$ Parity ({split_label})", fontsize=11, fontweight="bold",
                                 color=INK, loc="left", pad=10)
                _parity_panel(ax_kcat, pinn_predictions, pinn_run["kcat"]["overall"],
                             "log_kcat_pred", "log_kcat_true", r"$\log_{10}k_{cat}$", kcat_range)

                ax_km.set_title(rf"$K_m$ Parity ({split_label})", fontsize=11, fontweight="bold",
                               color=INK, loc="left", pad=10)
                _parity_panel(ax_km, pinn_predictions, pinn_run["km"]["overall"],
                             "log_km_pred", "log_km_true", r"$\log_{10}K_m$", km_range)

                ax_pct.set_title(f"Error Concentration ({split_label})", fontsize=11, fontweight="bold",
                                color=INK, loc="left", pad=10)
                _percentile_panel(ax_pct, pinn_run["kcat"]["percentile_bins"], pinn_run["km"]["percentile_bins"],
                                 pct_y_max)

        save(fig, filename)

    if pinn_specs:
        _draw_figure(pinn_specs, "results_summary_pinn")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2: R^2, wild-type vs. mutant (grouped), validation vs. benchmark (colored)
# ═══════════════════════════════════════════════════════════════════════════
def _r2_bar_panel(ax, val_run: dict, bench_run: dict, param: str, title: str, ylim: tuple) -> None:
    """Draw a grouped-bar panel of R^2, wild-type vs. mutant, validation vs. benchmark.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    val_run, bench_run : dict
        Evaluation records for the 'val' and 'benchmark' splits, as
        returned by `_get_run`.
    param : str
        "kcat" or "km".
    title : str
        Panel title.
    ylim : tuple of float
        Shared (lo, hi) y-axis range, so this panel plots on the same
        scale as its sibling parameter panel.

    Returns
    -------
    None
    """
    groups = ["Wild-Type", "Mutant"]
    group_keys = ["wild_type", "mutant"]
    val_r2 = [val_run[param].get(k, {}).get("r2", float("nan")) for k in group_keys]
    bench_r2 = [bench_run[param].get(k, {}).get("r2", float("nan")) for k in group_keys]

    x = np.arange(len(groups))
    width = 0.35
    bars_val = ax.bar(x - width / 2, val_r2, width, color=VALIDATION_COLOR, label="Validation")
    bars_bench = ax.bar(x + width / 2, bench_r2, width, color=BENCHMARK_COLOR, label="Benchmark")

    lo, hi = ylim
    headroom = (hi - lo) * 0.06
    ax.set_ylim(lo, hi)
    for bars in (bars_val, bars_bench):
        for bar in bars:
            v = bar.get_height()
            if np.isnan(v):
                continue
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, v + (headroom * 0.12 if v >= 0 else -headroom * 0.12),
                    f"{v:.3f}", ha="center", va=va, fontsize=8, color=INK)

    ax.set_xticks(x, groups, fontsize=9.5)
    ax.set_ylabel(r"$R^2$", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)
    ax.legend(loc="best", fontsize=8.5, frameon=False)


def generate_r2_comparison(store: dict, model_type: str) -> None:
    """Generate the kcat/Km R^2 comparison figure: wild-type vs. mutant
    (grouped), validation vs. benchmark (colored).

    Saves "results_r2_comparison" via `_style.save`.

    Parameters
    ----------
    store : dict
        Results store, as returned by `_load_eval_store`.
    model_type : str
        Model type to render.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If `store` is missing a 'val' or 'benchmark' run for `model_type`.
    """
    val_run = _get_run(store, "val", model_type)
    bench_run = _get_run(store, "benchmark", model_type)

    fig = _new_fig(11.0, 4.6)
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.07, right=0.97, top=0.86, bottom=0.14)
    ax_kcat = fig.add_subplot(gs[0, 0])
    ax_km = fig.add_subplot(gs[0, 1])
    for ax in (ax_kcat, ax_km):
        _style_axes(ax)
    for ax, letter in [(ax_kcat, "a"), (ax_km, "b")]:
        ax.text(-0.14, 1.14, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK)

    # R^2 is the same bounded metric in both panels, so both share one
    # y-axis - otherwise the bars would visually mislead about relative
    # kcat vs. Km model quality.
    all_vals = [
        run[param].get(k, {}).get("r2", float("nan"))
        for run in (val_run, bench_run) for param in ("kcat", "km") for k in ("wild_type", "mutant")
    ]
    all_vals = [v for v in all_vals if not np.isnan(v)]
    headroom = (max(abs(v) for v in all_vals) * 0.15 or 0.1) if all_vals else 0.1
    ylim = (min(0.0, min(all_vals, default=0.0) - headroom), max(all_vals, default=0.0) + headroom)

    _r2_bar_panel(ax_kcat, val_run, bench_run, "kcat", r"$k_{cat}$ $R^2$", ylim)
    _r2_bar_panel(ax_km, val_run, bench_run, "km", r"$K_m$ $R^2$", ylim)

    save(fig, "results_r2_comparison")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3: absolute log-error density, wild-type vs. mutant
# ═══════════════════════════════════════════════════════════════════════════
def _hist_density_max(errors_list: list, bins) -> float:
    """Peak density across several error histograms sharing `bins`.

    Parameters
    ----------
    errors_list : list of list of float
        One list of absolute log-errors per histogram (e.g. wild-type/
        mutant, validation/benchmark).
    bins : array_like
        Bin edges shared by every histogram, as passed to
        `numpy.histogram`.

    Returns
    -------
    float
        The maximum density value across all histograms, used as a shared
        y-axis ceiling so sibling panels plot on the same scale.
    """
    peak = 0.0
    for errors in errors_list:
        if errors:
            counts, _ = np.histogram(errors, bins=bins, density=True)
            peak = max(peak, counts.max())
    return peak


def _error_hist_panel(ax, param_summary: dict, bins, title: str, y_max: float) -> None:
    """Draw one absolute-log-error density histogram panel, wild-type vs. mutant.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    param_summary : dict
        Evaluation summary for one parameter/split, with
        "wild_type_abs_log_errors" and "mutant_abs_log_errors" lists.
    bins : array_like
        Bin edges shared with sibling panels, as passed to
        `matplotlib.axes.Axes.hist`.
    title : str
        Panel title.
    y_max : float
        Shared y-axis ceiling (density), from `_hist_density_max`.

    Returns
    -------
    None
    """
    wt_errors = param_summary.get("wild_type_abs_log_errors", [])
    mut_errors = param_summary.get("mutant_abs_log_errors", [])
    for errors, color, hatch, label in [
        (wt_errors, NAVY, None, "Wild-Type"),
        (mut_errors, MAGENTA, "///", "Mutant"),
    ]:
        if errors:
            ax.hist(errors, bins=bins, color=color, alpha=0.45, hatch=hatch,
                   edgecolor=color, linewidth=1.0, label=label, density=True)
    ax.set_ylim(0, y_max * 1.05)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax.set_xlabel(r"$|\Delta\log_{10}|$", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("density", fontsize=9, color=INK_SECONDARY)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)


def generate_error_histograms(store: dict, model_type: str) -> None:
    """Generate the kcat/Km absolute-log-error density histogram figure.

    Renders a 2x2 grid: one row per split ('val', 'benchmark'), one column
    per parameter (kcat, Km), each panel overlaying wild-type and mutant
    density histograms on shared per-parameter bin edges and y-axis scale.
    Saves "results_error_histograms" via `_style.save`.

    Parameters
    ----------
    store : dict
        Results store, as returned by `_load_eval_store`.
    model_type : str
        Model type to render.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If `store` is missing a 'val' or 'benchmark' run for `model_type`.
    """
    val_run = _get_run(store, "val", model_type)
    bench_run = _get_run(store, "benchmark", model_type)

    fig = _new_fig(11.5, 8.6)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28, left=0.08, right=0.97, top=0.90, bottom=0.07)
    axes = {(0, 0): fig.add_subplot(gs[0, 0]), (0, 1): fig.add_subplot(gs[0, 1]),
           (1, 0): fig.add_subplot(gs[1, 0]), (1, 1): fig.add_subplot(gs[1, 1])}
    for ax in axes.values():
        _style_axes(ax)
    for (row, col), letter in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ["a", "b", "c", "d"]):
        pos = gs[row, col].get_position(fig)
        fig.text(pos.x0 - 0.045, pos.y1 + 0.25 / fig.get_figheight(), letter,
                 fontsize=15, fontweight="bold", color=INK, ha="left", va="bottom")

    # shared bin edges per parameter (not across kcat/Km, whose error scales
    # can differ) so the validation/benchmark panels for the same parameter
    # are directly comparable at a glance.
    for col, (param, symbol) in enumerate([("kcat", r"$k_{cat}$"), ("km", r"$K_m$")]):
        combined = (val_run[param].get("wild_type_abs_log_errors", []) +
                   val_run[param].get("mutant_abs_log_errors", []) +
                   bench_run[param].get("wild_type_abs_log_errors", []) +
                   bench_run[param].get("mutant_abs_log_errors", []))
        hi = max(combined) if combined else 1.0
        bins = np.linspace(0, hi * 1.02, 40)
        y_max = _hist_density_max([
            val_run[param].get("wild_type_abs_log_errors", []),
            val_run[param].get("mutant_abs_log_errors", []),
            bench_run[param].get("wild_type_abs_log_errors", []),
            bench_run[param].get("mutant_abs_log_errors", []),
        ], bins)
        _error_hist_panel(axes[(0, col)], val_run[param], bins, f"Validation - {symbol}", y_max)
        _error_hist_panel(axes[(1, col)], bench_run[param], bins, f"Benchmark - {symbol}", y_max)

    save(fig, "results_error_histograms")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 4: sequence identity cutoffs
# ═══════════════════════════════════════════════════════════════════════════
def _seq_id_bar_panel(ax, bench_run: dict, param: str, title: str, ylim: tuple) -> None:
    """Draw an R^2-by-sequence-identity-cutoff bar panel for one parameter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    bench_run : dict
        Benchmark-split evaluation record, as returned by `_get_run`; must
        contain a "seq_id_cutoffs" key (an empty-data placeholder is drawn
        and the function returns early otherwise).
    param : str
        "kcat" or "km".
    title : str
        Panel title.
    ylim : tuple of float
        Shared (lo, hi) y-axis range, so this panel plots on the same
        scale as its sibling parameter panel.

    Returns
    -------
    None
    """
    cutoffs = ["seq_id_<=40", "seq_id_<=60", "seq_id_<=80", "seq_id_<=99"]
    labels = ["<= 40%", "<= 60%", "<= 80%", "<= 99%"]
    
    if "seq_id_cutoffs" not in bench_run:
        ax.text(0.5, 0.5, "No sequence ID data", ha="center", va="center", transform=ax.transAxes, color=INK_MUTED)
        return

    r2_vals = []
    for c in cutoffs:
        if c in bench_run["seq_id_cutoffs"]:
            r2 = bench_run["seq_id_cutoffs"][c][param].get("overall", {}).get("r2", float("nan"))
            r2_vals.append(r2)
        else:
            r2_vals.append(float("nan"))

    x = np.arange(len(labels))
    width = 0.55
    color = NAVY if param == "kcat" else MAGENTA
    bars = ax.bar(x, r2_vals, width, color=color)

    lo, hi = ylim
    headroom = (hi - lo) * 0.06
    ax.set_ylim(lo, hi)
    for bar in bars:
        v = bar.get_height()
        if np.isnan(v):
            continue
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, v + (headroom * 0.12 if v >= 0 else -headroom * 0.12),
                f"{v:.3f}", ha="center", va=va, fontsize=8, color=INK)

    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_ylabel(r"$R^2$", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlabel("Max Sequence Identity to Training Set", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)


def generate_seq_id_cutoffs(store: dict, model_type: str) -> None:
    """Generate the kcat/Km R^2-vs-sequence-identity-cutoff figure.

    Renders one panel per parameter (kcat, Km), each a bar chart of R^2 at
    increasing max training-set sequence identity cutoffs. Saves
    "results_seq_id_cutoffs_{model_type}" via `_style.save`; a no-op
    (logged) if the benchmark split has no run or no seq_id_cutoffs data
    for `model_type`.

    Parameters
    ----------
    store : dict
        Results store, as returned by `_load_eval_store`.
    model_type : str
        Model type to render.

    Returns
    -------
    None
    """
    try:
        bench_run = _get_run(store, "benchmark", model_type)
    except FileNotFoundError:
        return

    if "seq_id_cutoffs" not in bench_run:
        logger.info(f"No seq_id_cutoffs found for {model_type} benchmark run.")
        return

    fig = _new_fig(11.0, 4.6)
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.07, right=0.97, top=0.86, bottom=0.14)
    ax_kcat = fig.add_subplot(gs[0, 0])
    ax_km = fig.add_subplot(gs[0, 1])
    for ax in (ax_kcat, ax_km):
        _style_axes(ax)
    for ax, letter in [(ax_kcat, "a"), (ax_km, "b")]:
        ax.text(-0.14, 1.14, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK)

    cutoffs = ["seq_id_<=40", "seq_id_<=60", "seq_id_<=80", "seq_id_<=99"]
    all_vals = []
    for param in ("kcat", "km"):
        for c in cutoffs:
            if c in bench_run["seq_id_cutoffs"]:
                v = bench_run["seq_id_cutoffs"][c][param].get("overall", {}).get("r2", float("nan"))
                if not np.isnan(v):
                    all_vals.append(v)
                    
    headroom = (max(abs(v) for v in all_vals) * 0.15 or 0.1) if all_vals else 0.1
    ylim = (min(0.0, min(all_vals, default=0.0) - headroom), max(all_vals, default=1.0) + headroom)

    _seq_id_bar_panel(ax_kcat, bench_run, "kcat", r"$k_{cat}$ Sequence ID Cutoffs", ylim)
    _seq_id_bar_panel(ax_km, bench_run, "km", r"$K_m$ Sequence ID Cutoffs", ylim)

    save(fig, f"results_seq_id_cutoffs_{model_type}")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Parse CLI arguments, load the evaluation results store, and generate
    every results figure (parity/error-concentration, cross-model
    comparison, R^2 comparison, error histograms, sequence-identity
    cutoffs), skipping and logging any figure whose required split/model
    data is not yet present.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="ThermoKP Results Summary Figures")
    parser.add_argument("--eval_path", type=str, default=EVAL_PATH_DEFAULT,
                        help="Shared evaluation results store written by evaluate_dataset.py.")
    parser.add_argument("--model_type", type=str, default="pinn", choices=["pinn", "baseline", "dlkcat", "unikp", "catpred"])
    args = parser.parse_args()

    store = _load_eval_store(PROJECT_ROOT / args.eval_path)

    try:
        generate_results_summary(store, args.model_type)
    except FileNotFoundError as e:
        logger.warning(f"Skipping results_summary figures: {e}")
        
    try:
        generate_crossmodel_comparison(store)
    except Exception as e:
        logger.warning(f"Skipping results_crossmodel_comparison.svg: {e}")

    try:
        generate_r2_comparison(store, args.model_type)
    except FileNotFoundError as e:
        logger.warning(f"Skipping results_r2_comparison.svg: {e}")

    try:
        generate_error_histograms(store, args.model_type)
    except FileNotFoundError as e:
        logger.warning(f"Skipping results_error_histograms.svg: {e}")
        
    try:
        generate_seq_id_cutoffs(store, args.model_type)
    except FileNotFoundError as e:
        logger.warning(f"Skipping results_seq_id_cutoffs.svg: {e}")


if __name__ == "__main__":
    main()
