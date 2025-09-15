# aqp_engine/engine.py

import sqlite3
import time
import os
import re
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from .sampler import SampleManager
from .rewriter import QueryRewriter

class ApproxQueryEngine:
    """Orchestrates the AQP process."""

    def __init__(self, db_path: str, verbose: bool = True):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.sample_manager = SampleManager(self.conn, verbose)
        self.query_rewriter = QueryRewriter(self.conn)
        self.verbose = verbose

    def setup_samples(self, table_name: str, ratio: float, stratified_col: str, hash_col: str):
        """Creates all necessary sample tables for a given sampling ratio."""
        self.sample_manager.create_uniform_sample(table_name, ratio)
        self.sample_manager.create_stratified_sample(table_name, stratified_col, 50, ratio)
        self.sample_manager.create_hash_sample(table_name, hash_col, ratio)

    def populate_database_from_csv(self, csv_filename: str, table_name: str, index_col: Optional[str] = None):
        """Loads data from a CSV file into the database."""
        if self.verbose:
            print(f"--- Loading data from '{csv_filename}' into table '{table_name}' ---")

        if not os.path.exists(csv_filename):
            print(f"ERROR: Dataset file not found at '{csv_filename}'")
            print("Please make sure your CSV file is in the same folder as the script.")
            return False

        try:
            df = pd.read_csv(csv_filename)
            df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns]
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)

            if index_col and index_col in df.columns:
                if self.verbose:
                    print(f"Creating index on '{index_col}' column for performance...")
                self.cursor.execute(f'CREATE INDEX IF NOT EXISTS `idx_{index_col}` ON `{table_name}`(`{index_col}`);')

            self.conn.commit()
            if self.verbose:
                print("Database setup from CSV complete.\n")
            return True
        except Exception as e:
            print(f"An error occurred while loading the CSV: {e}")
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
                print(f"INFO: No suitable sample found.")
            return [], 0.0

        point_estimate_sql, aggregates, group_by_cols = self.query_rewriter.rewrite(sql_query, sample_table,
                                                                                    for_error_estimation=False)
        if not point_estimate_sql:
            if self.verbose:
                print(f"WARNING: Could not generate point estimate query.")
            return [], 0.0

        self.cursor.execute(point_estimate_sql)
        columns = [desc[0] for desc in self.cursor.description]
        point_estimate_results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]

        point_estimates_map = {}
        for row in point_estimate_results:
            group_key = tuple(row.get(k) for k in group_by_cols) if group_by_cols else ('full_table',)
            point_estimates_map[group_key] = row

        error_estimation_sql, _, _ = self.query_rewriter.rewrite(sql_query, sample_table, for_error_estimation=True)
        if not error_estimation_sql:
            if self.verbose:
                print(f"WARNING: Could not generate error estimation query.")
            return [], 0.0

        self.cursor.execute(error_estimation_sql)
        columns = [desc[0] for desc in self.cursor.description]
        subsample_results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]

        errors_by_group = {}
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

        final_results = []
        for group_key, agg_errors in errors_by_group.items():
            final_row = {}
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

                if not scaled_errors or point_estimate is None: continue

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