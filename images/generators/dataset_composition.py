"""Generates the dataset composition figure: EC-class / wild-type-vs-mutant
breakdown and kcat/Km value distributions queried live from
thermokp_database.db, plus a pipeline attrition summary parsed from the
ingestion/cleaning/validation/tensorization log files under data/results/
and data/*.txt.

The attrition panel's log inputs (VALIDATOR_SUMMARY_FILE, TENSORS_SUMMARY_FILE,
FAILED_*_FILE) may lag the pipeline stage they describe - each is read
defensively (missing/empty renders as "pending", never a misleading zero)
and every path is a named constant below so a moved file is a one-line
change. Validation/tensorization can also run on a remote worker whose
database and tensor cache are not this checkout's: the validated/tensorized
counts below come from those workers' logs, not from re-querying
clean_parameters or counting local .pt files, and the EC-class/value
panels stay on clean_parameters as currently checked out here (labeled
accordingly) since the validated subset itself isn't necessarily synced.

Run: python -m images.generators.dataset_composition
Output: images/generated/dataset_composition.svg
"""
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sankeyflow import Sankey

from ._style import GOOD, INK, INK_MUTED, INK_SECONDARY, MAGENTA, NAVY, SURFACE, save

# Set to True while tensor generation is actively running, so
# tensors_summary.txt/failed_tensors.txt (which describe an in-progress run,
# not a finished one) are treated as pending rather than read as final counts.
TENSORIZATION_RERUNNING = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "thermokp_database.db"

# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline log/artifact paths (mid-flight while validation/tensorization is
#  still running - keep these as the single place to repoint if a filename
#  or stage changes)
# ═══════════════════════════════════════════════════════════════════════════
RAW_SUMMARY_FILE = PROJECT_ROOT / "data" / "results" / "raw_summary.txt"
CLEAN_SUMMARY_FILE = PROJECT_ROOT / "data" / "results" / "clean_summary.txt"
VALIDATOR_SUMMARY_FILE = PROJECT_ROOT / "data" / "results" / "validator_summary.txt"
TENSORS_SUMMARY_FILE = PROJECT_ROOT / "data" / "results" / "tensors_summary.txt"
FAILED_CHEMICALS_FILE = PROJECT_ROOT / "data" / "failed_chemicals.txt"
FAILED_SEQUENCES_FILE = PROJECT_ROOT / "data" / "failed_sequences.txt"
FAILED_STRUCTURES_FILE = PROJECT_ROOT / "data" / "failed_structures.txt"
FAILED_TENSORS_FILE = PROJECT_ROOT / "data" / "failed_tensors.txt"
FAILED_BENCHMARK_FILE = PROJECT_ROOT / "data" / "failed_benchmark.txt"

# clean_records.py/generate_tensors.py log their drop reasons as prose, not
# "key : value" lines (unlike dataset_validator.py/the raw ingestion
# scripts) - one regex per known sentence template, easy to extend if a
# message changes.
CLEAN_SUMMARY_PATTERNS = {
    "variance_dropped": r"Dropped (\d+) records due to high intra-group variance",
    "duplicates_removed": r"removed (\d+) duplicates",
    "small_ec_class_dropped": r"Dropped (\d+) records belonging to top-level EC classes",
    "non_positive_dropped": r"Dropped (\d+) records with a non-positive kcat or Km",
    "benchmark_holdout": r"Set aside (\d+) records across (\d+) enzymes \((\d+) carrying mutants\)",
}
TENSORS_SUMMARY_PATTERNS = {
    "tensors_generated": r"Tensors generated\s*:\s*(\d+)\s*/\s*(\d+)",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


#  python-code-conventions.md's mandated logging.basicConfig format
#  ("%(asctime)s  %(levelname)-8s  %(message)s") prefixes every line a
#  pipeline stage writes via `logger.info` (raw_summary.txt is the one
#  exception, written by plain file I/O with no such prefix) - strip it
#  before matching so the "Key : value" summary block parses either way.
_LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+")


def _parse_keyvalue_summary(path: Path) -> dict:
    """Generic 'Key : integer' line parser, for the boxed summary blocks
    raw_summary.txt/validator_summary.txt print at the end of a run.

    Parameters
    ----------
    path : Path
        Summary log file to parse.

    Returns
    -------
    dict
        {key: int} for every "Key : integer" line found. Empty if the file
        is missing or empty (the stage has not run/finished yet).
    """
    out = {}
    for line in _read_text(path).splitlines():
        line = _LOG_PREFIX_RE.sub("", line)
        match = re.match(r"^([A-Za-z][\w /()\-]*?)\s*:\s*(-?\d+)\s*$", line)
        if match:
            out[match.group(1).strip()] = int(match.group(2))
    return out


def _parse_prose_summary(path: Path, patterns: dict) -> dict[str, Optional[tuple[int, ...]]]:
    """Extract one or more integer groups per named regex in `patterns` from
    a log that reports its numbers as prose sentences rather than "key :
    value" lines (clean_summary.txt, tensors_summary.txt).

    Parameters
    ----------
    path : Path
        Summary log file to parse.
    patterns : dict
        {key: regex} mapping; each regex's captured groups are read as
        integers.

    Returns
    -------
    dict
        {key: tuple of int} for every pattern that matched, or {key: None}
        for a pattern that found no match (the stage has not run/finished
        yet, or the sentence in question never fired for this run).
    """
    text = _read_text(path)
    out: dict[str, Optional[tuple[int, ...]]] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        out[key] = tuple(int(g) for g in match.groups()) if match else None
    return out


def _count_lines(path: Path) -> int:
    """Count non-blank lines in `path`, used to size a failure log's flow/count.

    Parameters
    ----------
    path : Path
        Log file to count (e.g. failed_tensors.txt, failed_benchmark.txt).

    Returns
    -------
    int
        Number of non-blank lines, or 0 if `path` does not exist.
    """
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# thermokp.ErrorCode values that appear in failed_benchmark.txt's "[TKP-NNN]"
# tags, given a human label here since the benchmark holdout is evaluated
# via thermokp.py's live inference pipeline (thermokp.ThermoKPError), not
# generate_tensors.py's failure path.
TKP_ERROR_LABELS = {
    1: "sequence unavailable",
    4: "mutation residue mismatch",
    6: "substrate unresolved",
    9: "structure pipeline failed",
}


def _count_tkp_errors(path: Path) -> dict:
    """Tally thermokp.ThermoKPError codes from a failed_benchmark.txt-style log.

    Parameters
    ----------
    path : Path
        Log file whose lines carry "[TKP-NNN]" tags (thermokp.ErrorCode
        values), typically failed_benchmark.txt.

    Returns
    -------
    dict
        {TKP code: occurrence count}.
    """
    counts: dict = {}
    for match in re.finditer(r"\[TKP-(\d+)\]", _read_text(path)):
        code = int(match.group(1))
        counts[code] = counts.get(code, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════
#  Database queries
# ═══════════════════════════════════════════════════════════════════════════
def _ec_class_breakdown(conn: sqlite3.Connection, table: str) -> list:
    """Wild-type/mutant record counts per top-level EC class.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to thermokp_database.db.
    table : str
        Table to query (e.g. "clean_parameters").

    Returns
    -------
    list of tuple
        (ec_top_level, wild_type_count, mutant_count) rows, ordered EC 1-6.
    """
    rows = conn.execute(f"""
        SELECT CAST(substr(ec_number, 1, instr(ec_number, '.') - 1) AS INTEGER) AS ec_top,
               SUM(CASE WHEN mutation IS NULL OR mutation = '' THEN 1 ELSE 0 END) AS wt,
               SUM(CASE WHEN mutation IS NOT NULL AND mutation != '' THEN 1 ELSE 0 END) AS mut
        FROM {table} GROUP BY ec_top ORDER BY ec_top
    """).fetchall()
    return rows


def _value_distributions(conn: sqlite3.Connection, table: str) -> dict:
    """Per-record log10(kcat)/log10(Km) values and their mutant/wild-type split.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to thermokp_database.db.
    table : str
        Table to query (e.g. "clean_parameters").

    Returns
    -------
    dict
        "log_kcat" and "log_km": 1-D `numpy.ndarray` of float, one entry
        per record. "is_mutant": 1-D `numpy.ndarray` of bool, aligned
        record-for-record with the two value arrays.
    """
    rows = conn.execute(f"""
        SELECT kcat, km, CASE WHEN mutation IS NOT NULL AND mutation != '' THEN 1 ELSE 0 END
        FROM {table}
    """).fetchall()
    kcat = np.log10(np.array([r[0] for r in rows], dtype=float))
    km = np.log10(np.array([r[1] for r in rows], dtype=float))
    is_mut = np.array([bool(r[2]) for r in rows], dtype=bool)
    return {"log_kcat": kcat, "log_km": km, "is_mutant": is_mut}


def _source_db_breakdown(conn: sqlite3.Connection, table: str) -> list:
    """Record counts per source database, descending by count.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to thermokp_database.db.
    table : str
        Table to query (e.g. "raw_parameters").

    Returns
    -------
    list of tuple
        (source_db, record_count) rows, descending by record_count.
    """
    return conn.execute(f"SELECT source_db, COUNT(*) FROM {table} GROUP BY source_db ORDER BY 2 DESC").fetchall()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Query thermokp_database.db and parse the pipeline logs, then render
    and save the three-panel dataset composition figure (EC-class
    breakdown, kcat/Km distributions, pipeline attrition Sankey).

    Returns
    -------
    None
    """
    conn = sqlite3.connect(DB_PATH)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(13.333, 12.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.28,
                          left=0.06, right=0.97, top=0.94, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b0 = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    for ax in (ax_a, ax_b0):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)

    # --- panel a: EC-class composition, wild-type vs mutant ---
    ax_a.text(-0.10, 1.14, "a", transform=ax_a.transAxes, fontsize=15, fontweight="bold", color=INK)
    ec_rows = _ec_class_breakdown(conn, "clean_parameters")
    labels = [f"EC {r[0]}" for r in ec_rows]
    wt_counts = np.array([r[1] for r in ec_rows], dtype=float)
    mut_counts = np.array([r[2] for r in ec_rows], dtype=float)
    y_pos = np.arange(len(labels))[::-1]

    ax_a.barh(y_pos, wt_counts, color=NAVY, label="Wild-Type", height=0.62)
    ax_a.barh(y_pos, mut_counts, left=wt_counts, color=MAGENTA, label="Mutant", height=0.62)
    totals = wt_counts + mut_counts
    for y, total, mut in zip(y_pos, totals, mut_counts):
        ax_a.text(total + totals.max() * 0.02, y, f"{int(total):,}  ({mut / total:.0%} mut.)",
                  va="center", ha="left", fontsize=8, color=INK_SECONDARY)
    ax_a.set_yticks(y_pos, labels)
    ax_a.set_xlabel("clean_parameters records", fontsize=9, color=INK_SECONDARY)
    ax_a.set_xlim(0, totals.max() * 1.30)
    ax_a.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax_a.set_title("Dataset Composition by EC Class", fontsize=11, fontweight="bold",
                   color=INK, loc="left", pad=10)

    # --- panel b: log10(kcat) / log10(Km) distributions ---
    ax_b0.text(-0.10, 1.14, "b", transform=ax_b0.transAxes, fontsize=15, fontweight="bold", color=INK)
    dist = _value_distributions(conn, "clean_parameters")
    ax_b0.set_title("kcat / Km Distributions", fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)
    bins = np.linspace(-6, 8, 45).tolist()
    for values, color, hatch, label in [
        (dist["log_kcat"], NAVY, None, r"$\log_{10} k_{cat}$"),
        (dist["log_km"], MAGENTA, "///", r"$\log_{10} K_m$"),
    ]:
        ax_b0.hist(values, bins=bins, color=color, alpha=0.45, hatch=hatch,
                  edgecolor=color, linewidth=1.0, label=label, density=True)
    ax_b0.set_xlabel(r"$\log_{10}$(value)  ·  $k_{cat}$ in s$^{-1}$, $K_m$ in mM", fontsize=9, color=INK_SECONDARY)
    ax_b0.set_ylabel("density", fontsize=9, color=INK_SECONDARY)
    ax_b0.legend(loc="upper right", fontsize=9, frameon=False)

    # --- panel c: pipeline attrition, as a Sankey (flow width = record count) ---
    ax_c.text(-0.020, 1.10, "c", transform=ax_c.transAxes, fontsize=15, fontweight="bold", color=INK)
    ax_c.set_title("Pipeline Attrition", fontsize=11, fontweight="bold", color=INK, loc="left", pad=18)
    ax_c.axis("off")

    raw_kv = _parse_keyvalue_summary(RAW_SUMMARY_FILE)
    clean_kv = _parse_prose_summary(CLEAN_SUMMARY_FILE, CLEAN_SUMMARY_PATTERNS)
    validator_kv = _parse_keyvalue_summary(VALIDATOR_SUMMARY_FILE)
    tensors_kv = _parse_prose_summary(TENSORS_SUMMARY_FILE, TENSORS_SUMMARY_PATTERNS)

    source_rows = dict(_source_db_breakdown(conn, "raw_parameters"))
    brenda_total = source_rows.get("BRENDA", 0)
    sabio_total = source_rows.get("SABIO-RK", 0)
    raw_total = raw_kv.get("Total raw rows") or (brenda_total + sabio_total)
    
    # clean_parameters holds the post-validation record count
    clean_total = conn.execute("SELECT COUNT(*) FROM clean_parameters").fetchone()[0]
    benchmark_total = conn.execute("SELECT COUNT(*) FROM benchmark_parameters").fetchone()[0]
    
    tensorization_ran = not TENSORIZATION_RERUNNING and tensors_kv.get("tensors_generated") is not None
    n_failed_tensors = _count_lines(FAILED_TENSORS_FILE)

    variance_n = (clean_kv["variance_dropped"] or (0,))[0]
    dup_n = (clean_kv["duplicates_removed"] or (0,))[0]
    small_ec_n = (clean_kv["small_ec_class_dropped"] or (0,))[0]
    non_positive_n = (clean_kv["non_positive_dropped"] or (0,))[0]
    aggregate_drop = variance_n + dup_n + small_ec_n + non_positive_n
    pre_split = raw_total - aggregate_drop
    pre_validation_total = pre_split - benchmark_total

    # ── node levels (sources -> sinks) and flows between them, by name ──
    # sinks (Aggregate/Validation Drops, Tensor Build Failures) are listed
    # first in their level so they render above the continuing main flow,
    # consistent top-to-bottom throughout the diagram.
    levels = [
        [("BRENDA", brenda_total, {"color": NAVY}), ("SABIO-RK", sabio_total, {"color": NAVY})],
        [("Raw Ingestion", raw_total, {"color": NAVY})],
        [("Aggregate Drops", aggregate_drop, {"color": INK_MUTED}),
         ("Deduplicated Records", pre_split, {"color": NAVY})],
        [("Benchmark Holdout Table", benchmark_total, {"color": MAGENTA, "label_pos": "top"}),
         ("Pre-Validation", pre_validation_total, {"color": NAVY})],
    ]
    flows = [
        ("BRENDA", "Raw Ingestion", brenda_total),
        ("SABIO-RK", "Raw Ingestion", sabio_total),
        ("Raw Ingestion", "Aggregate Drops", aggregate_drop),
        ("Raw Ingestion", "Deduplicated Records", pre_split),
        ("Deduplicated Records", "Benchmark Holdout Table", benchmark_total),
        ("Deduplicated Records", "Pre-Validation", pre_validation_total),
    ]

    # Benchmark Holdout Table is evaluated via thermokp.py's live inference
    # pipeline (evaluate_dataset.py), not generate_tensors.py - a handful of
    # entries fail that featurization too (failed_benchmark.txt), separately
    # from the main track's Tensor Build Failures below. Its two children
    # share a level with Clean Parameters Table's (Validation Drops/
    # Validated, or the pending placeholder) rather than getting their own:
    # sankeyflow allows a flow to skip levels (only dest_level > src_level is
    # enforced), and giving them their own level in between made the longer
    # Clean Parameters Table -> Validated band skip over that level and cut
    # diagonally across the Benchmark branch's column.
    n_failed_benchmark = _count_lines(FAILED_BENCHMARK_FILE)
    benchmark_evaluated = benchmark_total - n_failed_benchmark
    next_level = [("Benchmark Featurization Failures", n_failed_benchmark,
                  {"color": INK_MUTED, "label_pos": "top"}),
                 ("Benchmark Evaluated", benchmark_evaluated, {"color": GOOD, "label_pos": "top"})]
    flows += [
        ("Benchmark Holdout Table", "Benchmark Featurization Failures", n_failed_benchmark),
        ("Benchmark Holdout Table", "Benchmark Evaluated", benchmark_evaluated),
    ]

    if clean_total > 0:
        validation_drop = pre_validation_total - clean_total
        next_level += [("Validation Drops", validation_drop, {"color": INK_MUTED}),
                      ("Clean Parameters Table", clean_total, {"color": GOOD})]
        flows += [
            ("Pre-Validation", "Validation Drops", validation_drop),
            ("Pre-Validation", "Clean Parameters Table", clean_total),
        ]
        levels.append(next_level)
        tensor_source, tensor_source_total = "Clean Parameters Table", clean_total
    else:
        next_level.append(("Validation Pending", pre_validation_total, {"color": INK_MUTED}))
        flows.append(("Pre-Validation", "Validation Pending", pre_validation_total))
        levels.append(next_level)
        tensor_source, tensor_source_total = "Validation Pending", pre_validation_total
    benchmark_branch_level = len(levels) - 1

    # Forced to the pending branch while TENSORIZATION_RERUNNING (see top of
    # file); flip that flag once the re-run finishes to get real numbers back.
    if tensorization_ran:
        tensor_drop = n_failed_tensors
        tensorized_total = tensor_source_total - tensor_drop
        levels.append([("Tensor Build Failures", tensor_drop, {"color": INK_MUTED, "label_pos": "top"}),
                       ("Graph Tensors", tensorized_total, {"color": GOOD})])
        flows += [
            (tensor_source, "Tensor Build Failures", tensor_drop),
            (tensor_source, "Graph Tensors", tensorized_total),
        ]
    else:
        # Not a real count yet - {label}-only format avoids implying a
        # tensorized count before the re-run has actually produced one.
        levels.append([("Graph Tensors", tensor_source_total,
                       {"color": INK_MUTED, "label_format": "{label}\n(pending re-run)"})])
        flows.append((tensor_source, "Graph Tensors", tensor_source_total))

    # Extra vertical padding on the benchmark branch's level: its two thin
    # nodes (Benchmark Featurization Failures/Benchmark Evaluated) sit right
    # next to Validation Drops/Validated at the default padding and read as
    # cramped together - node_pad_y_{min,max} are per-level (not per-gap), so
    # this widens all three gaps in that level rather than just the first.
    node_pad_y_min = [0.01] * len(levels)
    node_pad_y_max = [0.05] * len(levels)
    node_pad_y_min[benchmark_branch_level] = 0.035
    node_pad_y_max[benchmark_branch_level] = 0.09

    sankey = Sankey(flows=flows, nodes=levels, align_y="top", flow_color_mode="dest",
                   node_pad_y_min=node_pad_y_min, node_pad_y_max=node_pad_y_max,
                   node_opts={"label_opts": {"fontsize": 8.3, "color": INK}})
    sankey.draw(ax=ax_c)

    # Drop-reason breakdowns and failure-log counts: sankeyflow's node layout
    # leaves no clear whitespace next to a node for supplementary text (labels
    # sit flush against densely-packed flow bands), so this detail goes in the
    # footnote below rather than overlapping the diagram itself.
    n_failed_chem = _count_lines(FAILED_CHEMICALS_FILE)
    n_failed_seq = _count_lines(FAILED_SEQUENCES_FILE)
    n_failed_struct = _count_lines(FAILED_STRUCTURES_FILE)
    footnote_lines = [
        f"Aggregate Drops ({aggregate_drop:,}): variance {variance_n:,} · duplicates merged {dup_n:,} · "
        f"small EC class {small_ec_n:,} · non-positive {non_positive_n:,}",
    ]
    if clean_total > 0:
        footnote_lines.append(
            f"Validation Drops ({pre_validation_total - clean_total:,}): "
            f"thermodynamics {validator_kv.get('Dropped (Thermodynamics)', '?'):,} · "
            f"chemicals {validator_kv.get('Dropped (Chemicals)', '?'):,} · "
            f"structures {validator_kv.get('Dropped (Structures)', '?'):,} · "
            f"sequence {validator_kv.get('Dropped (Sequence)', '?'):,}"
        )
    footnote_lines.append(
        f"Validation failure logs: {n_failed_chem} unresolved chemical names (failed_chemicals.txt) · "
        f"{n_failed_seq} failed (UniProt, mutation) pairs (failed_sequences.txt) · "
        f"{n_failed_struct} missing structures (failed_structures.txt)"
    )
    if n_failed_benchmark:
        tkp_counts = _count_tkp_errors(FAILED_BENCHMARK_FILE)
        tkp_breakdown = " · ".join(
            f"{TKP_ERROR_LABELS.get(code, f'TKP-{code:03d}')} {count}"
            for code, count in sorted(tkp_counts.items(), key=lambda kv: -kv[1])
        )
        footnote_lines.append(
            f"Benchmark Featurization Failures ({n_failed_benchmark:,}, failed_benchmark.txt, "
            f"thermokp.py live inference): {tkp_breakdown}"
    )
    if tensorization_ran:
        footnote_lines.append(
            f"Tensor Build Failures ({n_failed_tensors:,}, failed_tensors.txt): mostly 3D-conformer/pocket-"
            f"prediction failures (co-substrate conformer, pocket prediction, ligand conformer)"
        )
    ax_c_pos = ax_c.get_position()
    fig.text(ax_c_pos.x0, 0.015, "\n".join(footnote_lines), ha="left", va="bottom",
            fontsize=7.4, color=INK_MUTED, style="italic", linespacing=1.6)

    conn.close()
    save(fig, "dataset_composition")


if __name__ == "__main__":
    main()
