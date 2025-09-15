# aqp_engine/rewriter.py

import re
import math
from typing import Dict, List, Tuple, Optional

class QueryRewriter:
    """Parses and rewrites SQL for AQP, now with multi-aggregate support."""

    def __init__(self, conn, num_subsamples: int = 100):
        self.conn = conn
        self.num_subsamples = num_subsamples

    def rewrite(self, sql_query: str, sample_table_name: str, for_error_estimation: bool) -> Optional[
        Tuple[str, List[Dict[str, str]], List[str]]]:
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

        rewritten_selects = []
        original_aggregates = []

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