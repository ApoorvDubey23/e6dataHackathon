#!/usr/bin/env python3
import argparse
import os
import re
import time
import math
import sqlite3
import statistics
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# --- CONFIGURATION ---
DB_NAME = "hackathon.db"


def parse_ratios_str(r: str) -> List[float]:
    """
    Accepts '0.01,0.02, 5%, 10' (meaning 10%) etc. Returns [0.01, 0.02, 0.05, 0.10].
    Integers >=1 are treated as percentages. '5%' is 0.05. Trims spaces.
    """
    if not r:
        return [0.01, 0.02, 0.05, 0.10]
    out: List[float] = []
    for tok in r.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("%"):
            v = float(tok[:-1]) / 100.0
        else:
            v = float(tok)
            if v >= 1.0:
                v = v / 100.0
        if v <= 0 or v > 1:
            raise ValueError(f"Ratio '{tok}' must be in (0,1].")
        out.append(v)
    return out


class SampleManager:
    """Manages the creation and metadata of various sample types."""

    def __init__(self, conn, verbose: bool = True):
        self.conn = conn
        self.cursor = conn.cursor()
        self.samples_metadata: Dict[str, Dict[str, Any]] = {}
        self.verbose = verbose

    # ---------- helpers ----------
    def _table_columns(self, table: str) -> List[str]:
        info = self.cursor.execute(f"PRAGMA table_info(`{table}`)").fetchall()
        return [str(col[1]).strip() for col in info]

    def _has_column(self, table: str, col: str) -> bool:
        if not col:
            return False
        cols = self._table_columns(table)
        col_lc = col.strip().lower()
        return any(c.lower() == col_lc for c in cols)

    def _canonical_colname(self, table: str, col: str) -> Optional[str]:
        """Return the real column name as it exists in the table (post-sanitization), case-insensitive."""
        cols = self._table_columns(table)
        col_lc = (col or "").strip().lower()
        for c in cols:
            if c.lower() == col_lc:
                return c
        return None
    # -----------------------------

    def create_uniform_sample(self, base_table: str, ratio: float):
        sample_table_name = f"{base_table}_uniform_sample"
        if self.verbose:
            print(f"--- Creating {ratio * 100:.1f}% Uniform Sample for '{base_table}' ---", flush=True)
        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")
        self.cursor.execute(f"""
            CREATE TABLE `{sample_table_name}` AS
            SELECT *, {ratio} as sampling_prob FROM `{base_table}`
            WHERE (ABS(RANDOM()) / CAST(9223372036854775807 AS REAL)) < {ratio};
        """)
        self.conn.commit()
        sample_count = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
        if self.verbose:
            print(f"Uniform sample '{sample_table_name}' created with {sample_count:,} rows.\n", flush=True)
        self.samples_metadata[sample_table_name] = {
            'type': 'uniform', 'base_table': base_table, 'ratio': ratio
        }

    def _calculate_prob_for_stratified(self, n: int, m: int, delta: float = 0.001) -> float:
        """
        Probability to get at least 'm' samples from 'n' with confidence (1-delta),
        Normal approximation to Binomial (VerdictDB-style).
        """
        if m >= n:
            return 1.0
        z_delta = norm.ppf(delta)
        A = n * n + z_delta * z_delta * n
        B = -2 * m * n - z_delta * z_delta * n
        C = m * m
        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return 1.0
        p = (-B - math.sqrt(discriminant)) / (2 * A)
        return min(1.0, p)

    def create_stratified_sample(self, base_table: str, stratify_column: str,
                                 min_samples_per_stratum: int, ratio: float):
        sample_table_name = f"{base_table}_stratified_{stratify_column}"
        if self.verbose:
            print(f"--- Creating Stratified Sample on '{stratify_column}' ---", flush=True)

        # Validate column exists (post-sanitization)
        canon = self._canonical_colname(base_table, stratify_column)
        if canon is None:
            if self.verbose:
                print(f"WARN: Column '{stratify_column}' not found in '{base_table}'. Skipping stratified sample.")
            return
        stratify_column = canon

        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")

        if self.verbose:
            print("Pass 1: Calculating stratum sizes and probabilities...")
        strata_counts = self.cursor.execute(
            f"SELECT `{stratify_column}`, COUNT(*) FROM `{base_table}` GROUP BY `{stratify_column}`"
        ).fetchall()

        strata_probs: Dict[Any, float] = {}
        for stratum, count in strata_counts:
            prob = self._calculate_prob_for_stratified(n=count, m=min_samples_per_stratum)
            final_prob = max(ratio, prob)
            strata_probs[stratum] = final_prob
            if self.verbose:
                print(f"  - Stratum '{stratum}': count={count}, calculated_prob={final_prob:.4f}")

        if self.verbose:
            print("Pass 2: Building stratified sample...")
        base_table_info = self.cursor.execute(f"PRAGMA table_info(`{base_table}`)").fetchall()
        column_definitions = ", ".join([f'"{col[1]}" {col[2]}' for col in base_table_info])
        self.cursor.execute(f'CREATE TABLE `{sample_table_name}` ({column_definitions}, sampling_prob REAL)')

        base_columns_list = [f'`{col[1]}`' for col in base_table_info]
        base_columns = ", ".join(base_columns_list)
        insert_columns = ", ".join(base_columns_list + ['sampling_prob'])

        for stratum, prob in strata_probs.items():
            insert_sql = f"""
                INSERT INTO `{sample_table_name}` ({insert_columns})
                SELECT {base_columns}, {prob}
                FROM `{base_table}`
                WHERE `{stratify_column}` = ? 
                AND (ABS(RANDOM()) / CAST(9223372036854775807 AS REAL)) < {prob};
            """
            self.cursor.execute(insert_sql, (stratum,))

        self.conn.commit()
        sample_count = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
        if self.verbose:
            print(f"Stratified sample '{sample_table_name}' created with {sample_count:,} rows.\n")
        self.samples_metadata[sample_table_name] = {
            'type': 'stratified', 'base_table': base_table, 'column': stratify_column
        }

    def create_hash_sample(self, base_table: str, hash_column: str, ratio: float):
        """Creates a hashed (universe) sample."""
        if not hash_column:
            if self.verbose:
                print("INFO: No hash column provided. Skipping hash sample.")
            return

        # Validate column exists (post-sanitization)
        canon = self._canonical_colname(base_table, hash_column)
        if canon is None:
            if self.verbose:
                print(f"WARN: Column '{hash_column}' not found in '{base_table}'. Skipping hash sample.")
            return
        hash_column = canon

        sample_table_name = f"{base_table}_hash_{hash_column}"
        if self.verbose:
            print(f"--- Creating {ratio * 100:.1f}% Hashed Sample for '{base_table}' on column '{hash_column}' ---")
        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")

        self.cursor.execute(f"PRAGMA table_info(`{base_table}`)")
        col_type = ""
        for col in self.cursor.fetchall():
            if col[1] == hash_column:
                col_type = (col[2] or "").upper()
                break

        if 'INT' in col_type or 'REAL' in col_type or 'NUMERIC' in col_type:
            hash_expr = f"ABS(`{hash_column}`) % 100"
        else:
            hash_expr = (
                f"ABS(((INSTR(`{hash_column}`,'a')*31)"
                f"+(INSTR(`{hash_column}`,'e')*37)"
                f"+(LENGTH(`{hash_column}`)*41))) % 100"
            )

        self.cursor.execute(f"""
            CREATE TABLE `{sample_table_name}` AS
            SELECT *, {ratio} as sampling_prob FROM `{base_table}`
            WHERE ({hash_expr}) / 100.0 < {ratio};
        """)
        self.conn.commit()
        sample_count = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
        if self.verbose:
            print(f"Hashed sample '{sample_table_name}' created with {sample_count:,} rows.\n")
        self.samples_metadata[sample_table_name] = {
            'type': 'hash', 'base_table': base_table, 'column': hash_column, 'ratio': ratio
        }

    def find_best_sample_for_query(self, query: str) -> Optional[str]:
        """Intelligently selects the best sample table for a given query."""
        group_by_match = re.search(r"GROUP BY\s+([a-zA-Z0-9_`]+)", query, re.IGNORECASE | re.DOTALL)

        if group_by_match:
            group_by_col = group_by_match.group(1).strip().split(',')[0].strip().replace('`', '')
            for name, meta in self.samples_metadata.items():
                if meta.get('type') == 'stratified' and meta.get('column') == group_by_col:
                    if self.verbose:
                        print(f"INFO: Chose stratified sample '{name}' for GROUP BY query on '{group_by_col}'.")
                    return name

        for name, meta in self.samples_metadata.items():
            if meta.get('type') == 'uniform':
                if self.verbose:
                    print(f"INFO: Chose uniform sample '{name}' as a fallback.")
                return name

        return None

class QueryRewriter:
    """Parses and rewrites SQL for AQP, multi-aggregate + GROUP BY + subsampling for CIs."""

    def __init__(self, conn, num_subsamples: int = 100):
        self.conn = conn
        self.num_subsamples = num_subsamples

    def rewrite(self, sql_query: str, sample_table_name: str, for_error_estimation: bool) -> Optional[Tuple[str, List[Dict[str, str]], List[str]]]:
        select_pattern = re.compile(r"SELECT\s+(.*?)\s+FROM", re.IGNORECASE | re.DOTALL)
        groupby_pattern = re.compile(r"GROUP BY\s+(.*?)(?:ORDER BY|;|$)", re.IGNORECASE | re.DOTALL)

        select_match = select_pattern.search(sql_query)
        groupby_match = groupby_pattern.search(sql_query)
        if not select_match:
            return None, [], []

        select_clause = select_match.group(1).strip()
        group_by_clause = groupby_match.group(1).strip() if groupby_match else ""
        group_by_cols = [col.strip() for col in group_by_clause.split(',')] if group_by_clause else []

        n_cursor = self.conn.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`")
        n_result = n_cursor.fetchone()
        if not n_result or n_result[0] == 0:
            return None, [], []
        n = n_result[0]

        n_s = math.sqrt(n)
        prob_any_subsample = min(1.0, (self.num_subsamples * n_s) / n if n > 0 else 0)

        subsample_assignment_sql = f"""
            (CASE WHEN (ABS(RANDOM()) / 9223372036854775807.0) < {prob_any_subsample}
             THEN 1 + ABS(RANDOM()) % {self.num_subsamples} ELSE 0 END)
        """

        rewritten_selects: List[str] = []
        original_aggregates: List[Dict[str, str]] = []

        for part in select_clause.split(','):
            part = part.strip()
            if part in group_by_cols:
                continue
            agg_match = re.match(r"(SUM|AVG|COUNT)\((.*?)\)(?:\s+AS\s+(\w+))?", part, re.IGNORECASE)
            if agg_match:
                func, col, alias = agg_match.groups()
                alias = alias or f"{func.lower()}_{col.replace('*', 'all')}"
                original_aggregates.append({'func': func.upper(), 'alias': alias})

                if func.upper() == 'SUM':
                    expr = f"SUM({col} / sampling_prob) AS {alias}"
                elif func.upper() == 'COUNT':
                    expr = f"SUM(1.0 / sampling_prob) AS {alias}"
                elif func.upper() == 'AVG':
                    sum_alias = f"__v_sum_{alias}"
                    count_alias = f"__v_count_{alias}"
                    expr = f"SUM({col} / sampling_prob) AS {sum_alias}, SUM(1.0 / sampling_prob) AS {count_alias}"
                rewritten_selects.append(expr)

        if for_error_estimation:
            rewritten_selects.append("COUNT(*) as ns_i")

        rewritten_select_clause = ", ".join(rewritten_selects)
        group_by_statement = f"GROUP BY {group_by_clause}" if group_by_clause else ""

        if for_error_estimation:
            group_by_for_error = f"GROUP BY {group_by_clause}, subsample_id" if group_by_clause else "GROUP BY subsample_id"
            rewritten_query = f"""
                WITH variational_table AS (
                    SELECT *, {subsample_assignment_sql} as subsample_id FROM `{sample_table_name}`
                )
                SELECT {group_by_clause}{',' if group_by_clause else ''} subsample_id, {rewritten_select_clause}
                FROM variational_table
                WHERE subsample_id > 0
                {group_by_for_error}
            """
        else:
            rewritten_query = f"""
                SELECT {group_by_clause}{',' if group_by_clause else ''} {rewritten_select_clause}
                FROM `{sample_table_name}`
                {group_by_statement}
            """
        return rewritten_query, original_aggregates, group_by_cols


class ApproxQueryEngine:
    """Orchestrates the AQP process."""

    def __init__(self, db_path: str, verbose: bool = True):
        print(f"--- Initializing ApproxQueryEngine inside approxqueryEngine '{db_path}' ---", flush=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.sample_manager = SampleManager(self.conn, verbose)
        self.query_rewriter = QueryRewriter(self.conn)
        self.verbose = verbose

    def setup_samples(self, table_name: str, ratio: float, stratified_col: str, hash_col: str):
        """Creates all necessary sample tables for a given sampling ratio."""
        print(f"--- Setting up samples with ratio {ratio * 100:.1f}% ---", flush=True)
        self.sample_manager.create_uniform_sample(table_name, ratio)
        if stratified_col:
            self.sample_manager.create_stratified_sample(table_name, stratified_col, 50, ratio)
        if hash_col:
            self.sample_manager.create_hash_sample(table_name, hash_col, ratio)

    def populate_database_from_csv(self, csv_filename: str, table_name: str, index_col: Optional[str] = None):
        """Loads data from a CSV file into the database."""
        if self.verbose:
            print(f"--- Loading data from '{csv_filename}' into table '{table_name}' ---", flush=True)

        if not os.path.exists(csv_filename):
            print(f"ERROR: Dataset file not found at '{csv_filename}'", flush=True)
            print("Please make sure your CSV file is in the same folder as the script.", flush=True)
            return False

        try:
            df = pd.read_csv(csv_filename)
            # sanitize column names
            df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns]
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)

            if index_col and index_col in df.columns:
                if self.verbose:
                    print(f"Creating index on '{index_col}' column for performance...", flush=True)
                self.cursor.execute(f'CREATE INDEX IF NOT EXISTS `idx_{index_col}` ON `{table_name}`(`{index_col}`);')

            self.conn.commit()
            if self.verbose:
                print("Database setup from CSV complete.\n", flush=True)
            return True
        except Exception as e:
            print(f"An error occurred while loading the CSV: {e}", flush=True)
            return False

    def execute_exact_query(self, sql_query: str) -> Tuple[List[Dict], float]:
        start_time = time.time()
        self.cursor.execute(sql_query)
        columns = [desc[0] for desc in self.cursor.description]
        result = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        end_time = time.time()
        return result, end_time - start_time

    def execute_approximate_query(self, sql_query: str, confidence_level: float = 0.95) -> Tuple[List[Dict], float]:
        start_time = time.time()
        sample_table = self.sample_manager.find_best_sample_for_query(sql_query)
        if not sample_table:
            if self.verbose:
                print("INFO: No suitable sample found.")
            return [], 0.0

        point_estimate_sql, aggregates, group_by_cols = self.query_rewriter.rewrite(sql_query, sample_table, for_error_estimation=False)
        if not point_estimate_sql:
            if self.verbose:
                print("WARNING: Could not generate point estimate query.")
            return [], 0.0

        self.cursor.execute(point_estimate_sql)
        columns = [desc[0] for desc in self.cursor.description]
        point_estimate_results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]

        point_estimates_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for row in point_estimate_results:
            group_key = tuple(row.get(k) for k in group_by_cols) if group_by_cols else ('full_table',)
            point_estimates_map[group_key] = row

        error_estimation_sql, _, _ = self.query_rewriter.rewrite(sql_query, sample_table, for_error_estimation=True)
        if not error_estimation_sql:
            if self.verbose:
                print("WARNING: Could not generate error estimation query.")
            return [], 0.0

        self.cursor.execute(error_estimation_sql)
        columns = [desc[0] for desc in self.cursor.description]
        subsample_results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]

        errors_by_group: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
        n_total_sample = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table}`").fetchone()[0]
        ns_target = math.sqrt(n_total_sample) if n_total_sample > 0 else 1

        for row in subsample_results:
            group_key = tuple(row.get(k) for k in group_by_cols) if group_by_cols else ('full_table',)
            errors_by_group.setdefault(group_key, {agg['alias']: [] for agg in aggregates})

            point_estimate_row = point_estimates_map.get(group_key, {})
            ns_i = row.get('ns_i', 1)

            for agg_info in aggregates:
                alias = agg_info['alias']
                if agg_info['func'] == 'AVG':
                    sum_alias = f"__v_sum_{alias}"
                    count_alias = f"__v_count_{alias}"
                    subsample_estimate = row[sum_alias] / row[count_alias] if row.get(count_alias, 0) != 0 else 0
                    point_sum = point_estimate_row.get(sum_alias, 0)
                    point_count = point_estimate_row.get(count_alias, 0)
                    point_estimate = point_sum / point_count if point_count != 0 else 0
                else:
                    subsample_estimate = row.get(alias, 0)
                    point_estimate = point_estimate_row.get(alias, 0)

                scaled_error = math.sqrt(ns_i) * (subsample_estimate - point_estimate)
                errors_by_group[group_key][alias].append(scaled_error)

        final_results: List[Dict[str, Any]] = []
        for group_key, agg_errors in errors_by_group.items():
            final_row: Dict[str, Any] = {}
            if group_by_cols:
                for i, col_name in enumerate(group_by_cols):
                    final_row[col_name] = group_key[i]

            for agg_info in aggregates:
                alias = agg_info['alias']
                scaled_errors = agg_errors.get(alias, [])
                point_estimate_row = point_estimates_map.get(group_key, {})
                point_estimate = point_estimate_row.get(alias)
                if point_estimate is None and agg_info['func'] == 'AVG':
                    sum_alias = f"__v_sum_{alias}"
                    count_alias = f"__v_count_{alias}"
                    sum_val = point_estimate_row.get(sum_alias, 0)
                    count_val = point_estimate_row.get(count_alias, 0)
                    point_estimate = sum_val / count_val if count_val != 0 else 0

                if not scaled_errors or point_estimate is None:
                    continue

                lower_quantile = (1 - confidence_level) / 2
                upper_quantile = 1 - lower_quantile
                t_lower = np.quantile(scaled_errors, lower_quantile)
                t_upper = np.quantile(scaled_errors, upper_quantile)

                lower_bound = point_estimate - t_upper / math.sqrt(ns_target)
                upper_bound = point_estimate - t_lower / math.sqrt(ns_target)

                final_row[f"{alias}_estimate"] = point_estimate
                final_row[f"{alias}_lower"] = lower_bound
                final_row[f"{alias}_upper"] = upper_bound
            final_results.append(final_row)

        end_time = time.time()
        return final_results, end_time - start_time


def run_benchmark(engine: ApproxQueryEngine, query_name: str, sql_query: str, group_by_cols: List[str]):
    print(f"--- Benchmarking '{query_name}' ---", flush=True)
    print(f"Original SQL: {sql_query}", flush=True)

    exact_results, exact_time = engine.execute_exact_query(sql_query)
    approx_results, approx_time = engine.execute_approximate_query(sql_query)

    print(f"\nExact query took {exact_time:.4f}s.", flush=True)
    if not approx_results:
        print("Approximate query could not be run.\n", flush=True)
        return

    print(f"Approximate query with error estimation took {approx_time:.4f}s.", flush=True)
    speedup = exact_time / approx_time if approx_time > 0 else float('inf')
    print(f"\n=> Speedup: {speedup:.2f}x\n", flush=True)

    if not exact_results:
        print("No results from exact query to compare.", flush=True)
        return

    if not group_by_cols:
        exact_row = exact_results[0]
        approx_row = approx_results[0]
        agg_keys = list(exact_row.keys())

        print(f"--- Results ---", flush=True)
        header = f"{'Aggregate':<25} | {'Exact Value':>18} | {'Approx. Estimate':>20} | {'Confidence Interval':>40} | {'Actual Error':>15}"
        print(header, flush=True)
        print("-" * len(header), flush=True)

        for agg_key in agg_keys:
            exact_val = exact_row.get(agg_key)
            if exact_val is None:
                continue
            est_key = f"{agg_key}_estimate"
            if est_key not in approx_row:
                continue
            approx_val = approx_row[est_key]
            error = (abs(approx_val - exact_val) / exact_val) * 100 if exact_val != 0 else 0
            lower_key = f"{agg_key}_lower"
            upper_key = f"{agg_key}_upper"
            ci_str = f"[{approx_row.get(lower_key, 0):,.2f}, {approx_row.get(upper_key, 0):,.2f}]"
            print(f"{agg_key:<25} | {exact_val:18,.2f} | {approx_val:20,.2f} | {ci_str:>40} | {error:14.2f}%", flush=True)
        print()
        return

    agg_keys = [k for k in exact_results[0].keys() if k not in group_by_cols]
    exact_map = {tuple(row[k] for k in group_by_cols): row for row in exact_results}

    for agg_key in agg_keys:
        print(f"--- Results for '{agg_key}' ---", flush=True)
        header = "".join([f"{col:<15} | " for col in group_by_cols]) + f"{'Exact Value':>18} | {'Approx. Estimate':>20} | {'Confidence Interval':>40} | {'Actual Error':>15}"
        print(header, flush=True)
        print("-" * len(header), flush=True)

        all_errors = []
        approx_results.sort(key=lambda x: tuple(x.get(k, '') for k in group_by_cols))

        for approx_row in approx_results:
            group_val_tuple = tuple(approx_row.get(k) for k in group_by_cols)
            if any(v is None for v in group_val_tuple):
                continue
            exact_row = exact_map.get(group_val_tuple)
            if exact_row:
                exact_val = exact_row.get(agg_key)
                if exact_val is None:
                    continue
                est_key = f"{agg_key}_estimate"
                lower_key = f"{agg_key}_lower"
                upper_key = f"{agg_key}_upper"
                if est_key not in approx_row:
                    continue
                approx_val = approx_row[est_key]
                error = (abs(approx_val - exact_val) / exact_val) * 100 if exact_val != 0 else 0
                all_errors.append(error)
                ci_str = f"[{approx_row.get(lower_key, 0):,.2f}, {approx_row.get(upper_key, 0):,.2f}]"
                group_str = "".join([f"{str(v):<15} | " for v in group_val_tuple])
                print(f"{group_str}{exact_val:18,.2f} | {approx_val:20,.2f} | {ci_str:>40} | {error:14.2f}%", flush=True)

        avg_error = statistics.mean(all_errors) if all_errors else 0
        print(f"\n=> Average Actual Error for '{agg_key}': {avg_error:.2f}%\n", flush=True)


def run_groupby_example(engine: ApproxQueryEngine, table_name: str, group_col: str, numeric_col: str) -> Dict[str, Any]:
    """
    Run a GROUP BY example: COUNT(*) and AVG(numeric_col) grouped by group_col.
    Returns a dict with exact/approx rows + times + the SQL used.
    """
    sql = f"SELECT {group_col}, COUNT(*) AS trips, AVG({numeric_col}) AS avg_value FROM {table_name} GROUP BY {group_col}"
    exact_rows, t_exact = engine.execute_exact_query(sql)
    approx_rows, t_approx = engine.execute_approximate_query(sql)
    return {
        "sql": sql,
        "exact": {"rows": exact_rows, "time_sec": t_exact},
        "approx": {"rows": approx_rows, "time_sec": t_approx},
    }


def run_suite_to_json(
    dataset_filename: str = "yellow_tripdata_2015-01.csv",
    table_name: str = "trips",
    stratify_col: str = "RateCodeID",
    hash_col: str = "payment_type",
    numeric_col: str = "total_amount",
    ratios: List[float] = None,
    confidence: float = 0.95,
    verbose: bool = False,
    group_by_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the pipeline and return results as a JSON-serializable dict (safe for API).
    """
    ratios = ratios or [0.01, 0.02, 0.05, 0.10]
    print(f"--- Running AQP Benchmark with dataset '{dataset_filename}' into table '{table_name}' ---", flush=True)

    engine = ApproxQueryEngine(DB_NAME, verbose=verbose)
    try:
        ok = engine.populate_database_from_csv(
            csv_filename=dataset_filename,
            table_name=table_name,
            index_col=stratify_col,
        )
        if not ok:
            return {"ok": False, "error": f"Dataset '{dataset_filename}' not found or could not be loaded."}

        query1 = f"SELECT COUNT(*) AS total_trips, AVG({numeric_col}) as avg_revenue FROM {table_name}"
        query2 = f"SELECT SUM({numeric_col}) AS total_revenue FROM {table_name}"

        results_by_ratio: List[Dict[str, Any]] = []
        grouped_example_payload: Optional[Dict[str, Any]] = None

        for ratio in ratios:
            engine.setup_samples(table_name=table_name, ratio=ratio, stratified_col=stratify_col, hash_col=hash_col)

            exact1, t_exact1 = engine.execute_exact_query(query1)
            approx1, t_approx1 = engine.execute_approximate_query(query1, confidence_level=confidence)
            speedup1 = (t_exact1 / t_approx1) if t_approx1 > 0 else None

            exact2, t_exact2 = engine.execute_exact_query(query2)
            approx2, t_approx2 = engine.execute_approximate_query(query2, confidence_level=confidence)
            speedup2 = (t_exact2 / t_approx2) if t_approx2 > 0 else None

            results_by_ratio.append({
                "ratio": ratio,
                "queries": [
                    {
                        "name": "COUNT + AVG",
                        "sql": query1,
                        "exact": {"rows": exact1, "time_sec": t_exact1},
                        "approx": {"rows": approx1, "time_sec": t_approx1},
                        "speedup": speedup1,
                    },
                    {
                        "name": "SUM",
                        "sql": query2,
                        "exact": {"rows": exact2, "time_sec": t_exact2},
                        "approx": {"rows": approx2, "time_sec": t_approx2},
                        "speedup": speedup2,
                    }
                ]
            })

            # Run a single grouped example only once (on the first ratio) if requested
            if group_by_col and grouped_example_payload is None:
                grouped_example_payload = run_groupby_example(
                    engine=engine,
                    table_name=table_name,
                    group_col=group_by_col,
                    numeric_col=numeric_col,
                )

        return {
            "ok": True,
            "db_name": DB_NAME,
            "dataset": dataset_filename,
            "table": table_name,
            "stratify_col": stratify_col,
            "hash_col": hash_col,
            "numeric_col": numeric_col,
            "confidence": confidence,
            "results": results_by_ratio,
            "grouped_example": grouped_example_payload,  # may be None if not requested
        }
    finally:
        try:
            engine.conn.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="yellow_tripdata_2015-01.csv")
    p.add_argument("--table", default="trips")
    p.add_argument("--stratify-col", default="RateCodeID")
    p.add_argument("--hash-col", default="payment_type")
    p.add_argument("--numeric-col", default="total_amount")
    p.add_argument("--ratios", default="0.01,0.02,0.05,0.1", help="Comma-separated ratios, supports % or integers as percent")
    p.add_argument("--group-by", default="", help="Optional column to GROUP BY for a demo query")
    args = p.parse_args()

    ratios = parse_ratios_str(args.ratios)

    engine = ApproxQueryEngine(DB_NAME)
    is_data_loaded = engine.populate_database_from_csv(
        csv_filename=args.csv,
        table_name=args.table,
        index_col=args.stratify_col
    )
    if not is_data_loaded:
        return

    # Queries without GROUP BY
    query1 = f'SELECT COUNT(*) AS total_trips, AVG({args.numeric_col}) as avg_revenue FROM {args.table}'
    query2 = f'SELECT SUM({args.numeric_col}) AS total_revenue FROM {args.table}'

    for ratio in ratios:
        print("\n" + "=" * 80, flush=True)
        print(f"BENCHMARKING WITH SAMPLING RATIO: {ratio * 100:.1f}%", flush=True)
        print("=" * 80 + "\n", flush=True)

        engine.setup_samples(
            table_name=args.table,
            ratio=ratio,
            stratified_col=args.stratify_col,
            hash_col=args.hash_col
        )

        run_benchmark(
            engine=engine,
            query_name=f"Full Table COUNT and AVG ({ratio * 100:.1f}% sample)",
            sql_query=query1,
            group_by_cols=[]
        )
        run_benchmark(
            engine=engine,
            query_name=f"Full Table SUM ({ratio * 100:.1f}% sample)",
            sql_query=query2,
            group_by_cols=[]
        )

        # Optional: one GROUP BY demo
        if args.group_by:
            gb_sql = f"SELECT {args.group_by}, COUNT(*) AS trips, AVG({args.numeric_col}) AS avg_value FROM {args.table} GROUP BY {args.group_by}"
            run_benchmark(
                engine=engine,
                query_name=f"GROUP BY {args.group_by} ({ratio * 100:.1f}% sample)",
                sql_query=gb_sql,
                group_by_cols=[args.group_by]
            )


if __name__ == "__main__":
    main()
