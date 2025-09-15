import pandas as pd
import numpy as np
import pywt
import time
import sqlite3
import os
import re
import math
import statistics
from itertools import product
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# --- Configuration ---
DB_NAME = "aqp_benchmark.db"
DATASET_FILENAME = "yellow_tripdata_2015-01.csv"
TABLE_NAME = "trips"
WAVELET_DIMENSIONS = ["RateCodeID", "payment_type"]
COLUMN_TO_STRATIFY_ON = "RateCodeID"
COLUMN_FOR_HASH_SAMPLING = "payment_type"
NUMERIC_COLUMN_TO_AGGREGATE = "total_amount"
RATIOS_TO_TEST = [0.01, 0.05, 0.1]

# --- Global list to store benchmark results for plotting ---
benchmark_results = []


# --- ENGINE 1: Wavelet Approximate Query Engine ---
class AQPWaveletEngine:
    """
    An Approximate Query Processing engine using multi-dimensional Haar wavelets.
    Builds a compressed synopsis of the data for fast, approximate answers.
    """

    def __init__(self, df: pd.DataFrame, dimensions: List[str], measure: Dict[str, str]):
        self.dimensions = dimensions
        self.measure = measure
        self._prepare_data(df)
        self.synopsis = None
        self.coeffs_shape = None
        self.total_setup_time = 0

    def _prepare_data(self, df: pd.DataFrame):
        self.df = df.copy()
        self.dim_categories = {}
        for dim in self.dimensions:
            if dim not in self.df.columns:
                raise ValueError(f"Dimension '{dim}' not found in DataFrame columns.")
            self.df[dim] = self.df[dim].astype('category')
            self.dim_categories[dim] = self.df[dim].cat.categories

    def build_synopsis(self, compression_ratio: float = 0.1):
        print(f"Building Wavelet synopsis ({self.measure['agg']}) with {compression_ratio:.2%} compression...")
        start_time = time.time()
        if self.measure['agg'] == 'COUNT':
            grouped = self.df.groupby(self.dimensions, observed=False).size()
        else:
            grouped = self.df.groupby(self.dimensions, observed=False)[self.measure['column']].agg(
                self.measure['agg'].lower())
        all_combos = product(*(self.dim_categories[dim] for dim in self.dimensions))
        multi_index = pd.MultiIndex.from_tuples(all_combos, names=self.dimensions)
        self.data_cube = grouped.reindex(multi_index, fill_value=0).values
        self.data_cube = self.data_cube.reshape([len(cats) for cats in self.dim_categories.values()])
        coeffs = pywt.wavedecn(self.data_cube, 'haar')
        arr_coeffs, self.coeffs_shape = pywt.coeffs_to_array(coeffs)

        dc_component = arr_coeffs.flatten()[0]
        ac_coeffs = arr_coeffs.flatten()[1:]

        abs_coeffs = np.abs(ac_coeffs)
        k = int(len(abs_coeffs) * compression_ratio)
        if 0 < k < len(abs_coeffs):
            threshold = np.sort(abs_coeffs)[-k]
            ac_coeffs[np.abs(ac_coeffs) < threshold] = 0

        final_coeffs_flat = np.concatenate(([dc_component], ac_coeffs))
        self.synopsis = final_coeffs_flat.reshape(arr_coeffs.shape)

        self.total_setup_time = time.time() - start_time
        print(f"-> Synopsis built in {self.total_setup_time:.4f} seconds.")

    def _reconstruct_cube(self) -> np.ndarray:
        if self.synopsis is None:
            raise ValueError("Synopsis has not been built. Call build_synopsis() first.")
        coeffs_from_arr = pywt.array_to_coeffs(self.synopsis, self.coeffs_shape, output_format='wavedecn')
        return pywt.waverecn(coeffs_from_arr, 'haar')

    def query(self, group_by_cols: List[str] = []) -> Dict:
        approx_cube = self._reconstruct_cube()
        if not group_by_cols:
            return {'total': max(0, approx_cube.sum())}
        else:
            results = {}
            group_by_dim = group_by_cols[0]
            if group_by_dim not in self.dimensions:
                raise ValueError(f"Cannot group by '{group_by_dim}' as it's not a wavelet dimension.")
            dim_index = self.dimensions.index(group_by_dim)
            categories = self.dim_categories[group_by_dim]
            for i, category in enumerate(categories):
                slicer = [slice(None)] * len(self.dimensions)
                slicer[dim_index] = i
                results[category] = max(0, approx_cube[tuple(slicer)].sum())
            return results


# --- ENGINE 2: VerdictDB-style Sampling AQP Engine (Full Functionality) ---
class SampleManager:
    """Manages the creation and metadata of various sample types."""

    def __init__(self, conn, verbose: bool = True):
        self.conn = conn
        self.cursor = conn.cursor()
        self.samples_metadata = {}
        self.verbose = verbose
        self.total_setup_time = 0

    def create_uniform_sample(self, base_table: str, ratio: float):
        sample_table_name = f"{base_table}_uniform_sample_{int(ratio * 100)}"
        print(f"Creating {ratio:.2%} uniform sample '{sample_table_name}'...")
        start_time = time.time()
        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")
        self.cursor.execute(f"""
            CREATE TABLE `{sample_table_name}` AS
            SELECT *, {ratio} as sampling_prob FROM `{base_table}`
            WHERE (ABS(RANDOM()) / CAST(9223372036854775807 AS REAL)) < {ratio};
        """)
        self.conn.commit()
        self.samples_metadata[sample_table_name] = {'type': 'uniform', 'base_table': base_table, 'ratio': ratio}
        setup_time = time.time() - start_time
        self.total_setup_time += setup_time
        count = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
        print(f"-> Uniform sample created with {count:,} rows in {setup_time:.4f} seconds.")

    def create_stratified_sample(self, base_table: str, stratify_column: str, ratio: float, min_samples: int = 20):
        sample_table_name = f"{base_table}_stratified_{stratify_column}_{int(ratio * 100)}"
        print(f"Creating stratified sample on '{stratify_column}'...")
        start_time = time.time()
        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")
        strata_counts = self.cursor.execute(
            f"SELECT `{stratify_column}`, COUNT(*) FROM `{base_table}` GROUP BY `{stratify_column}`").fetchall()
        strata_probs = {stratum: max(ratio, self._calculate_prob(count, min_samples)) for stratum, count in
                        strata_counts}
        base_table_info = self.cursor.execute(f"PRAGMA table_info(`{base_table}`)").fetchall()
        column_definitions = ", ".join([f'"{col[1]}" {col[2]}' for col in base_table_info])
        self.cursor.execute(f'CREATE TABLE `{sample_table_name}` ({column_definitions}, sampling_prob REAL)')
        base_columns = ", ".join([f'`{col[1]}`' for col in base_table_info])
        insert_columns = f"{base_columns}, sampling_prob"
        for stratum, prob in strata_probs.items():
            self.cursor.execute(f"""
                INSERT INTO `{sample_table_name}` ({insert_columns})
                SELECT {base_columns}, {prob} FROM `{base_table}`
                WHERE `{stratify_column}` = ? AND (ABS(RANDOM()) / 9223372036854775807.0) < {prob};
            """, (stratum,))
        self.conn.commit()
        self.samples_metadata[sample_table_name] = {'type': 'stratified', 'base_table': base_table,
                                                    'column': stratify_column, 'ratio': ratio}
        setup_time = time.time() - start_time
        self.total_setup_time += setup_time
        count = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
        print(f"-> Stratified sample created with {count:,} rows in {setup_time:.4f} seconds.")

    def create_hash_sample(self, base_table: str, hash_column: str, ratio: float):
        sample_table_name = f"{base_table}_hash_{hash_column}_{int(ratio * 100)}"
        print(f"Creating {ratio:.2%} hash sample on '{hash_column}'...")
        start_time = time.time()
        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")
        hash_expr = f"ABS( ( (INSTR(`{hash_column}`, 'a') * 31) + (INSTR(`{hash_column}`, 'e') * 37) + (LENGTH(`{hash_column}`) * 41) ) ) % 100"
        self.cursor.execute(f"""
            CREATE TABLE `{sample_table_name}` AS
            SELECT *, {ratio} as sampling_prob FROM `{base_table}`
            WHERE ({hash_expr}) / 100.0 < {ratio};
        """)
        self.conn.commit()
        self.samples_metadata[sample_table_name] = {'type': 'hash', 'base_table': base_table, 'column': hash_column,
                                                    'ratio': ratio}
        setup_time = time.time() - start_time
        self.total_setup_time += setup_time
        count = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
        print(f"-> Hash sample created with {count:,} rows in {setup_time:.4f} seconds.")

    def _calculate_prob(self, n: int, m: int, delta: float = 0.001) -> float:
        if m >= n: return 1.0
        z_delta = norm.ppf(delta);
        A = n * n + z_delta * z_delta * n;
        B = -2 * m * n - z_delta * z_delta * n;
        C = m * m
        discriminant = B * B - 4 * A * C
        return min(1.0, (-B - math.sqrt(discriminant)) / (2 * A)) if discriminant >= 0 else 1.0

    def find_best_sample_for_query(self, query: str, ratio: float) -> Optional[str]:
        group_by_match = re.search(r"GROUP BY\s+([a-zA-Z0-9_`]+)", query, re.IGNORECASE)
        target_suffix = f"_{int(ratio * 100)}"
        if group_by_match:
            group_by_col = group_by_match.group(1).strip().replace('`', '')
            for name, meta in self.samples_metadata.items():
                if name.endswith(target_suffix) and meta.get('type') == 'stratified' and meta.get(
                        'column') == group_by_col:
                    print(f"INFO: Chose stratified sample '{name}' for query.")
                    return name
        for name, meta in self.samples_metadata.items():
            if name.endswith(target_suffix) and meta.get('type') == 'uniform':
                print(f"INFO: Chose uniform sample '{name}' for query.")
                return name
        return None


class QueryRewriter:
    def __init__(self, conn, num_subsamples: int = 100):
        self.conn = conn;
        self.num_subsamples = num_subsamples

    def rewrite(self, sql_query: str, sample_table_name: str, for_error_estimation: bool) -> Optional[
        Tuple[str, List[Dict[str, str]], List[str]]]:
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_query, re.IGNORECASE | re.DOTALL)
        groupby_match = re.search(r"GROUP BY\s+(.*?)(?:ORDER BY|;|$)", sql_query, re.IGNORECASE | re.DOTALL)
        if not select_match: return None, [], []
        select_clause = select_match.group(1).strip()
        group_by_clause = groupby_match.group(1).strip() if groupby_match else ""
        group_by_cols = [col.strip().replace('`', '') for col in group_by_clause.split(',')] if group_by_clause else []
        rewritten_selects, original_aggregates = [], []

        # Original parts from the select clause to handle GROUP BY columns
        original_select_parts = [p.strip() for p in select_clause.split(',')]

        for part in original_select_parts:
            # FIX: Use a more specific check to avoid incorrectly filtering aliases.
            # Only skip the part if it's an exact match for a group by column.
            if part in group_by_cols:
                continue

            agg_match = re.match(r"(SUM|AVG|COUNT)\((.*?)\)(?:\s+AS\s+([`\w]+))?", part, re.IGNORECASE)
            if agg_match:
                func, col, alias = agg_match.groups()
                alias = (alias or f"{func.lower()}_{col.replace('*', 'all').replace('`', '')}").replace('`', '')
                original_aggregates.append({'func': func.upper(), 'alias': alias})
                expr = f"SUM({col}/sampling_prob) AS {alias}" if func.upper() == 'SUM' else \
                    f"SUM(1.0/sampling_prob) AS {alias}" if func.upper() == 'COUNT' else \
                        f"SUM({col}/sampling_prob) AS __v_sum_{alias}, SUM(1.0/sampling_prob) AS __v_count_{alias}"
                rewritten_selects.append(expr)

        # Reconstruct the select list, making sure to include the group by columns
        final_select_list = []
        for p in original_select_parts:
            if p in group_by_cols:
                final_select_list.append(p)

        # Combine group by columns with the rewritten aggregates
        rewritten_select_clause = ", ".join(rewritten_selects)

        # Ensure there's a comma if both group by columns and aggregates are present
        if final_select_list and rewritten_select_clause:
            final_select_string = ", ".join(final_select_list) + ", " + rewritten_select_clause
        else:
            final_select_string = ", ".join(final_select_list) + rewritten_select_clause

        if for_error_estimation:
            n = self.conn.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
            if n == 0: return None, [], []
            n_s = math.sqrt(n)
            prob_any_subsample = min(1.0, (self.num_subsamples * n_s) / n if n > 0 else 0)
            subsample_sql = f"(CASE WHEN (ABS(RANDOM())/9223372036854775807.0)<{prob_any_subsample} THEN 1+ABS(RANDOM())%{self.num_subsamples} ELSE 0 END)"
            rewritten_selects.append("COUNT(*) as ns_i")
            # Rebuild the final select string for error estimation
            final_select_string_err = ", ".join(final_select_list) + (", " if final_select_list else "") + ", ".join(
                rewritten_selects)

            group_by_for_error = f"GROUP BY {group_by_clause}, subsample_id" if group_by_clause else "GROUP BY subsample_id"
            rewritten_query = f"WITH v AS (SELECT *, {subsample_sql} as subsample_id FROM `{sample_table_name}`) SELECT {group_by_clause}{',' if group_by_clause else ''} subsample_id, {', '.join(rewritten_selects)} FROM v WHERE subsample_id > 0 {group_by_for_error}"
        else:
            group_by_statement = f"GROUP BY {group_by_clause}" if group_by_clause else ""
            rewritten_query = f"SELECT {final_select_string} FROM `{sample_table_name}` {group_by_statement}"

        return rewritten_query, original_aggregates, group_by_cols


class SamplingAQPEngine:
    def __init__(self, db_path: str, verbose: bool = True):
        self.conn = sqlite3.connect(db_path)
        self.sample_manager = SampleManager(self.conn, verbose)
        self.query_rewriter = QueryRewriter(self.conn)
        self.total_setup_time = 0

    def setup_samples(self, table_name: str, ratio: float, stratified_col: str, hash_col: str):
        self.sample_manager.total_setup_time = 0
        self.sample_manager.create_uniform_sample(table_name, ratio)
        self.sample_manager.create_stratified_sample(table_name, stratified_col, ratio)
        self.sample_manager.create_hash_sample(table_name, hash_col, ratio)
        self.total_setup_time = self.sample_manager.total_setup_time

    def execute(self, sql_query: str, ratio: float, confidence_level: float = 0.95) -> List[Dict]:
        sample_table = self.sample_manager.find_best_sample_for_query(sql_query, ratio)
        if not sample_table: return []
        point_sql, aggregates, gb_cols = self.query_rewriter.rewrite(sql_query, sample_table, False)
        if not point_sql: return []
        point_cursor = self.conn.execute(point_sql)
        point_results = point_cursor.fetchall()
        point_cols = [d[0] for d in point_cursor.description] if point_cursor.description else []
        point_map = {tuple(row[point_cols.index(c)] for c in gb_cols): dict(zip(point_cols, row)) for row in
                     point_results} if gb_cols else \
            {('full_table',): dict(zip(point_cols, point_results[0])) if point_results else {}}
        error_sql, _, _ = self.query_rewriter.rewrite(sql_query, sample_table, True)
        if not error_sql: return [v for k, v in point_map.items()]
        error_cursor = self.conn.execute(error_sql)
        subsample_results = error_cursor.fetchall()
        subsample_cols = [d[0] for d in error_cursor.description] if error_cursor.description else []
        errors_by_group = {}
        n_total_sample = self.conn.execute(f"SELECT COUNT(*) FROM `{sample_table}`").fetchone()[0]
        ns_target = math.sqrt(n_total_sample) if n_total_sample > 0 else 1.0
        for row_tuple in subsample_results:
            row = dict(zip(subsample_cols, row_tuple))
            group_key = tuple(row.get(k) for k in gb_cols) if gb_cols else ('full_table',)
            errors_by_group.setdefault(group_key, {agg['alias']: [] for agg in aggregates})
            point_row = point_map.get(group_key, {})
            ns_i = row.get('ns_i', 1)
            for agg in aggregates:
                alias = agg['alias']
                if agg['func'] == 'AVG':
                    s_a, c_a = f"__v_sum_{alias}", f"__v_count_{alias}"
                    sub_est = row.get(s_a, 0) / row.get(c_a, 1) if row.get(c_a, 0) != 0 else 0
                    p_est = point_row.get(s_a, 0) / point_row.get(c_a, 1) if point_row.get(c_a, 0) != 0 else 0
                else:
                    sub_est = row.get(alias, 0);
                    p_est = point_row.get(alias, 0)
                errors_by_group[group_key][alias].append(math.sqrt(ns_i) * (sub_est - p_est))
        final_results = []
        for group_key, point_row in point_map.items():
            final_row = {col: key for col, key in zip(gb_cols, group_key)} if gb_cols else {}
            agg_errors = errors_by_group.get(group_key, {})
            for agg in aggregates:
                alias = agg['alias'];
                scaled_errors = agg_errors.get(alias, [])
                if agg['func'] == 'AVG':
                    s_a, c_a = f"__v_sum_{alias}", f"__v_count_{alias}"
                    point_est = point_row.get(s_a, 0) / point_row.get(c_a, 1) if point_row.get(c_a, 0) != 0 else 0
                else:
                    point_est = point_row.get(alias)
                final_row[f"{alias}_estimate"] = point_est
                if scaled_errors and point_est is not None:
                    t_l = np.quantile(scaled_errors, (1 - confidence_level) / 2)
                    t_u = np.quantile(scaled_errors, 1 - (1 - confidence_level) / 2)
                    final_row[f"{alias}_lower"] = point_est - t_u / ns_target
                    final_row[f"{alias}_upper"] = point_est - t_l / ns_target
                else:
                    final_row[f"{alias}_lower"] = point_est;
                    final_row[f"{alias}_upper"] = point_est
            final_results.append(final_row)
        return final_results

    def close(self):
        self.conn.close()


# --- ENGINE 3: Exact Query Engine (Ground Truth) ---
class SQLExactEngine:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path);
        self.cursor = self.conn.cursor()

    def execute(self, sql_query: str) -> List[Dict]:
        self.cursor.execute(sql_query)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def close(self): self.conn.close()


# --- BENCHMARKING SCRIPT ---
def run_comparison_benchmark(query_name: str, sql_query: str, group_by_cols: List[str], ratio: float):
    print("\n" + "=" * 120);
    print(f"BENCHMARKING QUERY: '{query_name}'");
    print(f"SQL: {sql_query}");
    print("=" * 120)
    start_sql = time.time();
    exact_results = sql_engine.execute(sql_query);
    sql_time = time.time() - start_sql
    if not exact_results: print("Could not get exact result. Aborting."); return
    start_sampling = time.time();
    sampling_results = sampling_engine.execute(sql_query, ratio);
    sampling_time = time.time() - start_sampling

    parsed_aggregates = re.findall(r"(COUNT|SUM|AVG)\((.*?)\)(?:\s+AS\s+([`\w]+))?", sql_query, re.IGNORECASE)

    start_wavelet = time.time()
    wavelet_results_by_agg = {}
    for agg_func, col, agg_alias in parsed_aggregates:
        agg_alias = (agg_alias or f"{agg_func.lower()}_{col.replace('*', 'all')}").replace('`', '')
        agg_func = agg_func.upper()
        if agg_func == 'AVG':
            sum_res = sum_engine.query(group_by_cols)
            count_res = count_engine.query(group_by_cols)
            wavelet_results_map = {}
            if not group_by_cols:
                wavelet_results_map[('full_table',)] = sum_res['total'] / count_res['total'] if count_res[
                                                                                                    'total'] > 0 else 0
            else:
                for group, total_sum in sum_res.items():
                    wavelet_results_map[group] = total_sum / count_res.get(group, 1) if count_res.get(group,
                                                                                                      0) > 0 else 0
            wavelet_results_by_agg[agg_alias] = wavelet_results_map
        else:
            wavelet_engine = count_engine if agg_func == 'COUNT' else sum_engine
            res = wavelet_engine.query(group_by_cols)
            wavelet_results_by_agg[agg_alias] = {k: v for k, v in res.items()} if group_by_cols else {
                ('full_table',): res.get('total', 0)}
    wavelet_time = time.time() - start_wavelet

    # Store timings for later plotting
    total_sampling_time = sampling_engine.total_setup_time + sampling_time
    total_wavelet_time = count_engine.total_setup_time + sum_engine.total_setup_time + wavelet_time
    benchmark_results.append({
        'query_name': query_name, 'ratio': ratio,
        'samp_setup_time': sampling_engine.total_setup_time, 'samp_query_time': sampling_time,
        'samp_total_time': total_sampling_time,
        'wav_setup_time': count_engine.total_setup_time + sum_engine.total_setup_time, 'wav_query_time': wavelet_time,
        'wav_total_time': total_wavelet_time,
        'exact_time': sql_time
    })

    print("\n--- Timings ---")
    header = f"{'Engine':<20} | {'Setup Time':>15} | {'Query Time':>15} | {'Total Time':>15} | {'Speedup vs Exact':>20}"
    print(header);
    print("-" * len(header))
    print(f"{'Exact SQL (SQLite)':<20} | {'N/A':>15} | {sql_time:15.4f}s | {sql_time:15.4f}s | {'1.00x':>20}")
    print(
        f"{'Approx. Sampling':<20} | {sampling_engine.total_setup_time:15.4f}s | {sampling_time:15.4f}s | {total_sampling_time:15.4f}s | {f'{sql_time / total_sampling_time:.2f}x':>20}")
    print(
        f"{'Approx. Wavelet':<20} | {(count_engine.total_setup_time + sum_engine.total_setup_time):15.4f}s | {wavelet_time:15.4f}s | {total_wavelet_time:15.4f}s | {f'{sql_time / total_wavelet_time:.2f}x':>20}")

    print("\n--- Accuracy Comparison ---")
    if not group_by_cols:
        exact_row = exact_results[0]
        sampling_row = sampling_results[0] if sampling_results else {}
        header = f"{'Aggregate':<20} | {'Exact Value':>18} | {'Sampling Est.':>18} | {'95% Confidence Interval':>35} | {'Wavelet Est.':>18} | {'Err %':>15}"
        print(header);
        print("-" * len(header))
        for agg_alias, exact_val in exact_row.items():
            sampling_val = sampling_row.get(f"{agg_alias}_estimate", 0)
            lower_b, upper_b = sampling_row.get(f"{agg_alias}_lower", 0), sampling_row.get(f"{agg_alias}_upper", 0)
            wavelet_val = wavelet_results_by_agg.get(agg_alias, {}).get(('full_table',), 0)
            sampling_err = (abs(sampling_val - exact_val) / exact_val) * 100 if exact_val != 0 else 0
            wavelet_err = (abs(wavelet_val - exact_val) / exact_val) * 100 if exact_val != 0 else 0
            print(
                f"{agg_alias:<20} | {exact_val:18,.2f} | {sampling_val:18,.2f} | {f'[{lower_b:,.2f}, {upper_b:,.2f}]':>35} | {wavelet_val:18,.2f} | {f'S:{sampling_err:4.2f}/W:{wavelet_err:4.2f}':>15}")
            # Store accuracy results for plotting
            benchmark_results[-1][f'{agg_alias}_samp_err'] = sampling_err
            benchmark_results[-1][f'{agg_alias}_wav_err'] = wavelet_err
    else:
        group_col = group_by_cols[0]
        exact_map = {row[group_col]: row for row in exact_results}
        sampling_map = {row.get(group_col): row for row in sampling_results} if sampling_results else {}
        agg_aliases = [a[2] or f"{a[0].lower()}_{a[1].replace('*', 'all')}".replace('`', '') for a in parsed_aggregates]
        header_p1 = f"{group_col:<15} | " + " | ".join([f"{alias:>18}" for alias in agg_aliases])
        header_p2 = " | ".join(
            [f"{'Samp Est':>18} | {'CI (Samp)':>35} | {'Wav Est':>18} | {'Err% (S/W)':>15}" for _ in agg_aliases])
        print(header_p1 + " | " + header_p2);
        print("-" * (len(header_p1) + 3 + len(header_p2)))

        all_errors = {alias: {'s': [], 'w': []} for alias in agg_aliases}
        for group_val, exact_row in sorted(exact_map.items()):
            sampling_row = sampling_map.get(group_val, {})
            line_str = f"{str(group_val):<15} | "
            exact_vals_str = " | ".join([f"{exact_row[alias]:18,.2f}" for alias in agg_aliases])
            line_str += exact_vals_str + " | "
            approx_vals_str = ""
            for alias in agg_aliases:
                exact_val = exact_row[alias]
                sampling_val = sampling_row.get(f"{alias}_estimate", 0)
                lower_b, upper_b = sampling_row.get(f"{alias}_lower", 0), sampling_row.get(f"{alias}_upper", 0)
                wavelet_val = wavelet_results_by_agg.get(alias, {}).get(group_val, 0)
                sampling_err = (abs(sampling_val - exact_val) / exact_val) * 100 if exact_val != 0 else 0
                wavelet_err = (abs(wavelet_val - exact_val) / exact_val) * 100 if exact_val != 0 else 0
                all_errors[alias]['s'].append(sampling_err);
                all_errors[alias]['w'].append(wavelet_err)
                approx_vals_str += f"{sampling_val:18,.2f} | {f'[{lower_b:,.2f}, {upper_b:,.2f}]':>35} | {wavelet_val:18,.2f} | {f'S:{sampling_err:4.2f}/W:{wavelet_err:4.2f}':>15} | "
            print(line_str + approx_vals_str.strip(" | "))

        print("-" * (len(header_p1) + 3 + len(header_p2)))
        avg_line = f"{'AVERAGE ERROR %':<15} | " + " | ".join([" " * 18 for _ in agg_aliases]) + " | "
        avg_err_str = ""
        for alias in agg_aliases:
            avg_s = statistics.mean(all_errors[alias]['s']) if all_errors[alias]['s'] else 0
            avg_w = statistics.mean(all_errors[alias]['w']) if all_errors[alias]['w'] else 0
            avg_err_str += f"{'':18} | {'':35} | {'':18} | {f'S:{avg_s:4.2f}/W:{avg_w:4.2f}':>15} | "
            # Store average accuracy results for plotting
            benchmark_results[-1][f'{alias}_avg_samp_err'] = avg_s
            benchmark_results[-1][f'{alias}_avg_wav_err'] = avg_w
        print(avg_line + avg_err_str.strip(" | "))


# --- PLOTTING FUNCTION ---
def generate_graphs(results: List[Dict]):
    """Generates and saves graphs from the benchmark results."""
    print("\n--- Generating Graphs ---")
    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results)

    # --- Plot 1: Speed Comparison (at a representative ratio) ---
    ratio_to_plot = 0.05
    speed_data = df[(df['ratio'] == ratio_to_plot) & (df['query_name'] == 'Full Table Aggregates')]
    if not speed_data.empty:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = ['Approx. Sampling', 'Approx. Wavelet']
        setup_times = [speed_data['samp_setup_time'].values[0], speed_data['wav_setup_time'].values[0]]
        query_times = [speed_data['samp_query_time'].values[0], speed_data['wav_query_time'].values[0]]

        bar_width = 0.35
        index = np.arange(len(labels))

        bar1 = ax.bar(index, setup_times, bar_width, label='Setup Time', color='skyblue')
        bar2 = ax.bar(index, query_times, bar_width, label='Query Time', bottom=setup_times, color='lightcoral')

        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Performance Comparison at {ratio_to_plot:.0%} Ratio/Compression')
        ax.set_xticks(index)
        ax.set_xticklabels(labels)
        ax.legend()

        for i, (s, q) in enumerate(zip(setup_times, query_times)):
            ax.text(i, s / 2, f'{s:.3f}s', ha='center', va='center', color='black', fontweight='bold')
            ax.text(i, s + q / 2, f'{q:.3f}s', ha='center', va='center', color='black', fontweight='bold')

        fig.tight_layout()
        plt.savefig('performance_comparison.png')
        print("Saved 'performance_comparison.png'")
        plt.close()

    # --- Plot 2: Accuracy vs. Ratio for SUM (Full Table) ---
    acc_data = df[df['query_name'] == 'Full Table Aggregates']
    if not acc_data.empty:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        ratios = acc_data['ratio'] * 100
        ax.plot(ratios, acc_data['total_revenue_samp_err'], 'o-', label='Sampling Error', color='royalblue')
        ax.plot(ratios, acc_data['total_revenue_wav_err'], 's-', label='Wavelet Error', color='firebrick')

        ax.set_xlabel('Synopsis Size (% of Original Data)')
        ax.set_ylabel('Average Relative Error (%)')
        ax.set_title('Accuracy for Full Table SUM (total_revenue)')
        ax.set_xticks(ratios)
        ax.legend()

        fig.tight_layout()
        plt.savefig('accuracy_full_table_sum.png')
        print("Saved 'accuracy_full_table_sum.png'")
        plt.close()

    # --- Plot 3: Accuracy vs. Ratio for COUNT (Full Table) ---
    if 'total_trips_samp_err' in acc_data.columns:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        ratios = acc_data['ratio'] * 100
        ax.plot(ratios, acc_data['total_trips_samp_err'], 'o-', label='Sampling Error', color='royalblue')
        ax.plot(ratios, acc_data['total_trips_wav_err'], 's-', label='Wavelet Error', color='firebrick')

        ax.set_xlabel('Synopsis Size (% of Original Data)')
        ax.set_ylabel('Average Relative Error (%)')
        ax.set_title('Accuracy for Full Table COUNT (total_trips)')
        ax.set_xticks(ratios)
        ax.legend()

        fig.tight_layout()
        plt.savefig('accuracy_full_table_count.png')
        print("Saved 'accuracy_full_table_count.png'")
        plt.close()

    # --- Plot 4: Accuracy vs. Ratio for AVG (Full Table) ---
    if 'avg_revenue_samp_err' in acc_data.columns:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        ratios = acc_data['ratio'] * 100
        ax.plot(ratios, acc_data['avg_revenue_samp_err'], 'o-', label='Sampling Error', color='royalblue')
        ax.plot(ratios, acc_data['avg_revenue_wav_err'], 's-', label='Wavelet Error', color='firebrick')

        ax.set_xlabel('Synopsis Size (% of Original Data)')
        ax.set_ylabel('Average Relative Error (%)')
        ax.set_title('Accuracy for Full Table AVG (avg_revenue)')
        ax.set_xticks(ratios)
        ax.legend()

        fig.tight_layout()
        plt.savefig('accuracy_full_table_avg.png')
        print("Saved 'accuracy_full_table_avg.png'")
        plt.close()

    # --- Plot 5: Accuracy for Grouped Query by RateCodeID (at a representative ratio) ---
    group_data = df[(df['ratio'] == ratio_to_plot) & (df['query_name'] == 'Grouped Aggregates by RateCodeID')]
    if not group_data.empty:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = ['revenue_by_rate', 'trips_by_rate']
        sampling_errors = [group_data['revenue_by_rate_avg_samp_err'].values[0],
                           group_data['trips_by_rate_avg_samp_err'].values[0]]
        wavelet_errors = [group_data['revenue_by_rate_avg_wav_err'].values[0],
                          group_data['trips_by_rate_avg_wav_err'].values[0]]

        index = np.arange(len(labels))
        bar_width = 0.35

        bar1 = ax.bar(index - bar_width / 2, sampling_errors, bar_width, label='Sampling Avg. Error',
                      color='mediumseagreen')
        bar2 = ax.bar(index + bar_width / 2, wavelet_errors, bar_width, label='Wavelet Avg. Error', color='indianred')

        ax.set_ylabel('Average Relative Error (%)')
        ax.set_title(f'Accuracy for Grouped Aggregates (by RateCodeID) at {ratio_to_plot:.0%} Ratio')
        ax.set_xticks(index)
        ax.set_xticklabels(labels)
        ax.legend()

        for i, (s, w) in enumerate(zip(sampling_errors, wavelet_errors)):
            ax.text(i - bar_width / 2, s + 0.5, f'{s:.2f}%', ha='center', va='bottom')
            ax.text(i + bar_width / 2, w + 0.5, f'{w:.2f}%', ha='center', va='bottom')

        fig.tight_layout()
        plt.savefig('accuracy_grouped_ratecode.png')
        print("Saved 'accuracy_grouped_ratecode.png'")
        plt.close()

    # --- Plot 6: Accuracy for Second Grouped Query by Payment Type (at a representative ratio) ---
    group_data_pmt = df[(df['ratio'] == ratio_to_plot) & (df['query_name'] == 'Grouped Aggregates by Payment Type')]
    if not group_data_pmt.empty:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = ['revenue_by_payment_type', 'trips_by_payment_type']
        sampling_errors = [group_data_pmt['revenue_by_payment_type_avg_samp_err'].values[0],
                           group_data_pmt['trips_by_payment_type_avg_samp_err'].values[0]]
        wavelet_errors = [group_data_pmt['revenue_by_payment_type_avg_wav_err'].values[0],
                          group_data_pmt['trips_by_payment_type_avg_wav_err'].values[0]]

        index = np.arange(len(labels))
        bar_width = 0.35

        bar1 = ax.bar(index - bar_width / 2, sampling_errors, bar_width, label='Sampling Avg. Error',
                      color='mediumseagreen')
        bar2 = ax.bar(index + bar_width / 2, wavelet_errors, bar_width, label='Wavelet Avg. Error', color='indianred')

        ax.set_ylabel('Average Relative Error (%)')
        ax.set_title(f'Accuracy for Grouped Aggregates (by Payment Type) at {ratio_to_plot:.0%} Ratio')
        ax.set_xticks(index)
        ax.set_xticklabels(labels)
        ax.legend()

        for i, (s, w) in enumerate(zip(sampling_errors, wavelet_errors)):
            ax.text(i - bar_width / 2, s + 0.5, f'{s:.2f}%', ha='center', va='bottom')
            ax.text(i + bar_width / 2, w + 0.5, f'{w:.2f}%', ha='center', va='bottom')

        fig.tight_layout()
        plt.savefig('accuracy_grouped_payment_type.png')
        print("Saved 'accuracy_grouped_payment_type.png'")
        plt.close()


if __name__ == "__main__":
    print("--- Starting AQP Benchmark: Wavelets vs. Sampling (Full Functionality) ---")
    if not os.path.exists(DATASET_FILENAME):
        print(f"FATAL ERROR: Dataset file not found at '{DATASET_FILENAME}'")
    else:
        # Load and prepare data
        # Ensure the full dataset is loaded by removing nrows
        print(f"Loading full dataset from '{DATASET_FILENAME}'... (This may take a while)")
        df = pd.read_csv(DATASET_FILENAME)
        print("Dataset loaded.")

        df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns]
        df = df[df[NUMERIC_COLUMN_TO_AGGREGATE] > 0].copy()
        for dim in WAVELET_DIMENSIONS:
            df[dim] = pd.to_numeric(df[dim], errors='coerce').fillna(0).astype(int)

        # Setup database for sampling engine
        if os.path.exists(DB_NAME): os.remove(DB_NAME)
        conn_init = sqlite3.connect(DB_NAME)
        print("Writing full dataset to SQLite database... (This may also take a while)")
        df.to_sql(TABLE_NAME, conn_init, if_exists='replace', index=False)
        conn_init.close()
        print(f"Data loaded into SQLite DB '{DB_NAME}'.")

        # Initialize engines
        sql_engine = SQLExactEngine(DB_NAME)
        sampling_engine = SamplingAQPEngine(DB_NAME)
        count_engine = AQPWaveletEngine(df, WAVELET_DIMENSIONS, {'agg': 'COUNT'})
        sum_engine = AQPWaveletEngine(df, WAVELET_DIMENSIONS, {'column': NUMERIC_COLUMN_TO_AGGREGATE, 'agg': 'SUM'})

        # Define benchmark queries
        query1 = f"SELECT SUM({NUMERIC_COLUMN_TO_AGGREGATE}) AS total_revenue, COUNT(*) AS total_trips, AVG({NUMERIC_COLUMN_TO_AGGREGATE}) as avg_revenue FROM {TABLE_NAME}"
        query2 = f"SELECT RateCodeID, SUM({NUMERIC_COLUMN_TO_AGGREGATE}) AS revenue_by_rate, COUNT(*) as trips_by_rate FROM {TABLE_NAME} GROUP BY RateCodeID"
        query3 = f"SELECT payment_type, SUM({NUMERIC_COLUMN_TO_AGGREGATE}) AS revenue_by_payment_type, COUNT(*) as trips_by_payment_type FROM {TABLE_NAME} GROUP BY payment_type"

        # Run benchmarks for each ratio
        for ratio in RATIOS_TO_TEST:
            print("\n" + "#" * 120);
            print(f"##  RUNNING BENCHMARK FOR RATIO: {ratio:.2%}");
            print("#" * 120)
            sampling_engine.setup_samples(TABLE_NAME, ratio, stratified_col=COLUMN_TO_STRATIFY_ON,
                                          hash_col=COLUMN_FOR_HASH_SAMPLING)
            count_engine.build_synopsis(compression_ratio=ratio)
            sum_engine.build_synopsis(compression_ratio=ratio)
            run_comparison_benchmark("Full Table Aggregates", query1, [], ratio=ratio)
            run_comparison_benchmark("Grouped Aggregates by RateCodeID", query2, ["RateCodeID"], ratio=ratio)
            run_comparison_benchmark("Grouped Aggregates by Payment Type", query3, ["payment_type"], ratio=ratio)

        # Generate and save plots after all benchmarks are complete
        generate_graphs(benchmark_results)

        # Clean up
        sql_engine.close();
        sampling_engine.close()
        if os.path.exists(DB_NAME): os.remove(DB_NAME)
        print("\n--- Benchmark Complete ---")

