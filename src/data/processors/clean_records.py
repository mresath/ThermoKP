"""
===========================================================================
Clean Records
Description: ThermoKP Database Cleanup & Aggregator
===========================================================================

Workflow:
1. Connect to `data/thermokp_database.db`.
2. Load `raw_parameters` into a pandas DataFrame.
3. Drop any rows still missing a `uniprot_id` (both parsers already attempt to
   resolve one at parse time - BRENDA via `fetch_uniprot_id`, SABIO-RK via the
   same function reused as a fallback - so this should be rare in practice).
4. Group the DataFrame by `[ec_number, substrate, co_substrates, uniprot_id,
   temperature, pH, mutation]`.
5. Reject (in full) any group whose kcat or Km values span more than
   `VARIANCE_RATIO_THRESHOLD`x between their max and min - an intentional
   safeguard against unit errors or unrecorded assay contaminants, not an
   attempt to smooth over ordinary replicate noise.
6. Compute the median of `kcat` and `km` ignoring NaN values for the
   surviving groups.
7. Drop aggregated records whose kcat or Km is non-positive - undefined in
   the model's log-space objective.
8. Hold out a reproducible batch of whole enzymes (see `_split_benchmark_set`)
   into a separate `benchmark_parameters` table for later model evaluation -
   removed from the training table entirely, not just sampled alongside it.
9. Write the remaining cleanly aggregated, unique rows to `clean_parameters`.

Known Caveats:
- Primary keys (`entry_id`) are completely regenerated during this step from 1 to N, destroying the original source ID mappings from SABIO/BRENDA.
- The benchmark holdout is chosen by `uniprot_id`, not by individual row - an
  enzyme is either entirely in `clean_parameters` or entirely in
  `benchmark_parameters`, so the benchmark measures generalization to
  never-seen proteins rather than just held-out measurements.

Author: ThermoKP Team
License: MIT
"""

from __future__ import annotations

import logging
import sqlite3
import pathlib

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  Logging configuration
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH: pathlib.Path = _PROJECT_ROOT / "data" / "thermokp_database.db"

# A replicate group (same enzyme/substrate/condition/mutation) is rejected in
# full if its kcat or Km values span more than this max/min ratio - an
# intentional safeguard against unit errors or unrecorded assay contaminants
# (e.g. an uncontrolled cofactor inflating kcat 10x), not a smoothing filter
# for ordinary experimental replicate noise. 5.0x keeps rejections rare: a
# stricter threshold rejects a large majority of multi-measurement groups in
# full, which is far more aggressive than the noise/contamination cases this
# filter is meant to catch.
VARIANCE_RATIO_THRESHOLD: float = 5.0

# Fixed seed so the benchmark holdout is reproducible across reruns - the
# whole point of a benchmark set is that results stay comparable over time,
# which requires holding out the *same* enzymes every time the pipeline runs.
BENCHMARK_RANDOM_SEED: int = 42
# Target number of distinct enzymes (uniprot_id) held out into
# `benchmark_parameters`. Capped at run time to a fraction of the dataset so
# it can't swallow an unreasonable share of a small dataset.
BENCHMARK_ENZYME_COUNT: int = 50
# Of the held-out enzymes, at least this many must carry at least one
# mutant record (non-null `mutation`) - a benchmark of only wild-type
# enzymes couldn't probe mutant sensitivity at all.
BENCHMARK_MIN_MUTANT_ENZYMES: int = 10


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregator Logic
def _enforce_class_representation(df: pd.DataFrame, min_samples: int = 100) -> pd.DataFrame:
    """Filter out any top-level EC class (e.g. EC 1, EC 2) that does not have at least `min_samples` records.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing aggregated enzyme kinetic records.
    min_samples : int, default: 100
        The minimum number of records a top-level EC class must have to be retained.
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only records from EC classes with >= min_samples records.
    """
    # Extract the top-level class (first digit of ec_number)
    top_level_class = df["ec_number"].astype(str).str.split('.').str[0]
    
    ec_counts = top_level_class.value_counts()
    valid_classes = [k for k, v in ec_counts.items() if v >= min_samples]
    
    filtered_df = pd.DataFrame(df[top_level_class.isin(valid_classes)].copy())
    
    dropped = len(df) - len(filtered_df)
    if dropped > 0:
        logger.info("Dropped %d records belonging to top-level EC classes with < %d samples.", dropped, min_samples)
        
    return filtered_df


def _split_benchmark_set(
    df: pd.DataFrame,
    n_enzymes: int = BENCHMARK_ENZYME_COUNT,
    min_mutant_enzymes: int = BENCHMARK_MIN_MUTANT_ENZYMES,
    random_seed: int = BENCHMARK_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a reproducible batch of whole enzymes for a benchmark set.

    Held-out enzymes (identified by `uniprot_id`) are removed from `df` in
    full - not just some of their rows - so the benchmark measures
    generalization to never-seen proteins, matching how the trained model
    will actually be evaluated. A subset of the held-out enzymes are
    guaranteed to carry at least one conservative point-mutant record (not
    just wild-type), so the benchmark can also probe mutant sensitivity.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned, aggregated (but not yet entry_id-assigned) records.
    n_enzymes : int, default: BENCHMARK_ENZYME_COUNT
        Target number of distinct `uniprot_id`s to hold out. Capped to a
        fraction of the available enzymes so it can't swallow a small
        dataset.
    min_mutant_enzymes : int, default: BENCHMARK_MIN_MUTANT_ENZYMES
        Minimum number of held-out enzymes that must carry at least one
        mutant record, capped to however many are actually available.
    random_seed : int, default: BENCHMARK_RANDOM_SEED
        Fixed seed so the same enzymes are held out across reruns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(remaining_df, benchmark_df)``.
    """
    rng = np.random.RandomState(random_seed)
    unique_enzymes: list = list(df["uniprot_id"].dropna().unique())

    # Never hold out more than a fifth of the dataset's enzymes.
    n_enzymes = min(n_enzymes, max(1, len(unique_enzymes) // 5))

    mutant_enzymes: list = list(df.loc[df["mutation"].notna(), "uniprot_id"].dropna().unique())
    n_mutant = min(min_mutant_enzymes, len(mutant_enzymes), n_enzymes)
    if n_mutant < min_mutant_enzymes:
        logger.warning(
            "Only %d mutant-carrying enzymes available (wanted %d) for the benchmark set.",
            n_mutant, min_mutant_enzymes,
        )

    chosen_mutant: list = list(rng.choice(mutant_enzymes, size=n_mutant, replace=False)) if n_mutant > 0 else []

    remaining_pool: list = [e for e in unique_enzymes if e not in set(chosen_mutant)]
    n_rest = min(max(0, n_enzymes - len(chosen_mutant)), len(remaining_pool))
    chosen_rest: list = list(rng.choice(remaining_pool, size=n_rest, replace=False)) if n_rest > 0 else []

    chosen: list = chosen_mutant + chosen_rest

    benchmark_mask = df["uniprot_id"].isin(chosen)
    benchmark_df = pd.DataFrame(df[benchmark_mask].copy())
    remaining_df = pd.DataFrame(df[~benchmark_mask].copy())

    return remaining_df, benchmark_df


def _clean_and_aggregate_database(db_path: pathlib.Path = DEFAULT_DB_PATH) -> None:
    """Read, clean, aggregate, and overwrite the SQLite database.
    
    Parameters
    ----------
    db_path : pathlib.Path, default: DEFAULT_DB_PATH
        The path to the SQLite database file.
        
    Returns
    -------
    None
    """
    if not db_path.exists():
        logger.error("Database not found at %s. Run parsers first.", db_path)
        return

    logger.info("Connecting to %s", db_path)
    conn = sqlite3.connect(db_path)
    
    # Read raw records
    df = pd.read_sql_query("SELECT * FROM raw_parameters", conn)
    initial_count = len(df)
    logger.info("Loaded %d raw records from the database.", initial_count)
    
    if initial_count == 0:
        logger.warning("Table is empty. Nothing to aggregate.")
        conn.close()
        return

    # Drop any rows that STILL have no UniProt ID (isoforms ambiguous or missing organism)
    df_clean = df.dropna(subset=["uniprot_id"]).copy()
    cleaned_count = len(df_clean)
    if cleaned_count < initial_count:
        logger.info("Dropped %d records due to missing/ambiguous UniProt IDs.", initial_count - cleaned_count)





    # 2. Pandas groupby drops None/NaN values in grouping keys by default.
    # We explicitly use dropna=False to group null values together.
    # `mutation` is included so a wild-type group never merges with a point
    # mutant's group, even for the same enzyme/substrate/condition.
    grouping_keys = ["ec_number", "measured_substrate", "co_substrates", "uniprot_id", "temperature", "ph", "mutation"]
    grouped = df_clean.groupby(grouping_keys, dropna=False)

    def filter_high_variance(g: pd.DataFrame) -> bool:
        """
        Filter out groups where the ratio between max and min value of
        kinetic parameters exceeds `VARIANCE_RATIO_THRESHOLD`.

        Parameters
        ----------
        g : pd.DataFrame
            The grouped dataframe.

        Returns
        -------
        bool
            True if the group should be kept, False if it has high variance.
        """
        for col in ["kcat", "km"]:
            vals = g[col].dropna()
            if len(vals) > 1:
                if vals.max() / (vals.min() + 1e-9) > VARIANCE_RATIO_THRESHOLD:
                    return False
        return True

    df_filtered = grouped.filter(filter_high_variance)
    filtered_count = len(df_filtered)
    if filtered_count < cleaned_count:
        logger.info("Dropped %d records due to high intra-group variance (max/min > %.1fx).",
                    cleaned_count - filtered_count, VARIANCE_RATIO_THRESHOLD)
        
    grouped = df_filtered.groupby(grouping_keys, dropna=False)
    # 3. Aggregate taking the median for kinetic parameters
    agg_df = grouped.agg({
        "kcat": "median",
        "km": "median"
    }).reset_index()
    
    final_count = len(agg_df)
    logger.info("Aggregated into %d unique clean pairs (removed %d duplicates).",
                final_count, filtered_count - final_count)
    
    # Keep first source_db and calculate group size
    source_df = df_filtered.groupby(grouping_keys, dropna=False)["source_db"].first().reset_index()
    size_series = pd.Series(df_filtered.groupby(grouping_keys, dropna=False).size())
    size_df = size_series.reset_index(name="group_size")
    
    agg_df = pd.merge(agg_df, source_df, on=grouping_keys)
    agg_df = pd.merge(agg_df, size_df, on=grouping_keys)
    
    # Set source_db to AGGREGATE only for rows  that were merged from multiple entries
    agg_df.loc[agg_df["group_size"] > 1, "source_db"] = "AGGREGATE"
    # Filter classes that have < 100 samples
    agg_df = _enforce_class_representation(agg_df, min_samples=100)

    # Drop records whose aggregated kcat or Km is non-positive. The training
    # objective is a log-space fit (see PINNMultiTaskLoss), where a zero or
    # negative rate constant has no defined target - a measured kcat of 0
    # (e.g. a catalytically dead mutant) is biologically real but cannot be
    # regressed in log10 space. NaN rows are preserved (the tensor pipeline
    # gates on non-null kcat/Km separately).
    positive_mask = ~(
        (agg_df["kcat"].notna() & (agg_df["kcat"] <= 0))
        | (agg_df["km"].notna() & (agg_df["km"] <= 0))
    )
    n_nonpositive = (~positive_mask).sum()
    if n_nonpositive:
        logger.info("Dropped %d records with a non-positive kcat or Km (undefined in log-space).", n_nonpositive)
    agg_df = pd.DataFrame(agg_df[positive_mask].copy())

    # Hold out a reproducible batch of whole enzymes for later benchmarking,
    # before entry_ids are assigned - removed from the training table
    # entirely so no held-out enzyme leaks into `clean_parameters`.
    agg_df, benchmark_df = _split_benchmark_set(agg_df)
    n_benchmark_mutant_enzymes = benchmark_df.loc[benchmark_df["mutation"].notna(), "uniprot_id"].nunique()
    logger.info(
        "Set aside %d records across %d enzymes (%d carrying mutants) into benchmark_parameters.",
        len(benchmark_df), benchmark_df["uniprot_id"].nunique(), n_benchmark_mutant_enzymes,
    )

    final_count = len(agg_df)

    # Generate fresh primary keys (independently for each table)
    agg_df["entry_id"] = range(1, final_count + 1)
    benchmark_df["entry_id"] = range(1, len(benchmark_df) + 1)

    # Reorder columns to match the standard schema
    schema_cols = ["entry_id", "source_db", "ec_number", "uniprot_id", "measured_substrate", "co_substrates", "kcat", "km", "temperature", "ph", "mutation"]
    agg_df = agg_df[schema_cols]
    benchmark_df = benchmark_df[schema_cols]

    # Write aggregated data to a clean_parameters table, and the held-out
    # enzymes to a separate benchmark_parameters table.
    agg_df.to_sql("clean_parameters", conn, if_exists="replace", index=False)
    benchmark_df.to_sql("benchmark_parameters", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    logger.info("Database cleanly overwritten with %d cleaned & aggregated records.", final_count)

# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Execute the database cleaning and aggregation pipeline.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    None
    """
    logger.info("============================================================")
    logger.info("ThermoKP — Database Cleanup & Aggregator")
    logger.info("============================================================")
    
    _clean_and_aggregate_database()
    
    logger.info("==========================================================================")
    logger.info("==                   Cleanup & Aggregation Summary                      ==")
    logger.info("==========================================================================")
    logger.info("Status                    : Complete")
    logger.info("==========================================================================")

if __name__ == "__main__":
    main()
