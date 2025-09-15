# aqp_engine/sampler.py

import math
import re
from scipy.stats import norm
from typing import Optional

class SampleManager:
    """Manages the creation and metadata of various sample types."""

    def __init__(self, conn, verbose: bool = True):
        """
        Initializes the SampleManager.

        Args:
            conn: An active sqlite3 connection object.
            verbose: If True, prints detailed logs during sample creation.
        """
        self.conn = conn
        self.cursor = conn.cursor()
        self.samples_metadata = {}
        self.verbose = verbose

    def create_uniform_sample(self, base_table: str, ratio: float):
        sample_table_name = f"{base_table}_uniform_sample"
        if self.verbose:
            print(f"--- Creating {ratio * 100:.1f}% Uniform Sample for '{base_table}' ---")
        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")
        self.cursor.execute(f"""
            CREATE TABLE `{sample_table_name}` AS
            SELECT *, {ratio} as sampling_prob FROM `{base_table}`
            WHERE (ABS(RANDOM()) / CAST(9223372036854775807 AS REAL)) < {ratio};
        """)
        self.conn.commit()
        sample_count = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
        if self.verbose:
            print(f"Uniform sample '{sample_table_name}' created with {sample_count:,} rows.\n")
        self.samples_metadata[sample_table_name] = {'type': 'uniform', 'base_table': base_table, 'ratio': ratio}

    def _calculate_prob_for_stratified(self, n: int, m: int, delta: float = 0.001) -> float:
        """
        Calculates sampling probability to get at least 'm' samples from 'n' items
        with a confidence of (1 - delta).
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

    def create_stratified_sample(self, base_table: str, stratify_column: str, min_samples_per_stratum: int,
                                 ratio: float):
        sample_table_name = f"{base_table}_stratified_{stratify_column}"
        if self.verbose:
            print(f"--- Creating Stratified Sample on '{stratify_column}' ---")
        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")

        if self.verbose:
            print("Pass 1: Calculating stratum sizes and probabilities...")
        strata_counts = self.cursor.execute(
            f"SELECT `{stratify_column}`, COUNT(*) FROM `{base_table}` GROUP BY `{stratify_column}`").fetchall()

        strata_probs = {}
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
        self.samples_metadata[sample_table_name] = {'type': 'stratified', 'base_table': base_table,
                                                    'column': stratify_column}

    def create_hash_sample(self, base_table: str, hash_column: str, ratio: float):
        """Creates a hashed (universe) sample."""
        sample_table_name = f"{base_table}_hash_{hash_column}"
        if self.verbose:
            print(f"--- Creating {ratio * 100:.1f}% Hashed Sample for '{base_table}' on column '{hash_column}' ---")
        self.cursor.execute(f"DROP TABLE IF EXISTS `{sample_table_name}`")

        self.cursor.execute(f"PRAGMA table_info(`{base_table}`)")
        col_type = ""
        for col in self.cursor.fetchall():
            if col[1] == hash_column:
                col_type = col[2].upper()
                break

        if 'INT' in col_type or 'REAL' in col_type or 'NUMERIC' in col_type:
            hash_expr = f"ABS(`{hash_column}`) % 100"
        else:
            hash_expr = f"ABS( ( (INSTR(`{hash_column}`, 'a') * 31) + (INSTR(`{hash_column}`, 'e') * 37) + (LENGTH(`{hash_column}`) * 41) ) ) % 100"

        self.cursor.execute(f"""
            CREATE TABLE `{sample_table_name}` AS
            SELECT *, {ratio} as sampling_prob FROM `{base_table}`
            WHERE ({hash_expr}) / 100.0 < {ratio};
        """)
        self.conn.commit()
        sample_count = self.cursor.execute(f"SELECT COUNT(*) FROM `{sample_table_name}`").fetchone()[0]
        if self.verbose:
            print(f"Hashed sample '{sample_table_name}' created with {sample_count:,} rows.\n")
        self.samples_metadata[sample_table_name] = {'type': 'hash', 'base_table': base_table, 'column': hash_column,
                                                    'ratio': ratio}

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