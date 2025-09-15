# run_benchmark.py

import statistics
import matplotlib.pyplot as plt
from typing import Dict, List

# Import the main class from our new package
from aqp_engine import ApproxQueryEngine

# --- CONFIGURATION ---
DB_NAME = "hackathon.db"

def run_benchmark(engine: ApproxQueryEngine, query_name: str, sql_query: str, group_by_cols: List[str]):
    print(f"--- Benchmarking '{query_name}' ---")
    if engine.verbose:
        print(f"Original SQL: {sql_query}")

    exact_results, exact_time = engine.execute_exact_query(sql_query)
    approx_results, approx_time = engine.execute_approximate_query(sql_query)

    print(f"\nExact query took {exact_time:.4f}s.")
    if not approx_results:
        print("Approximate query could not be run.\n")
        return {}

    print(f"Approximate query with error estimation took {approx_time:.4f}s.")
    speedup = exact_time / approx_time if approx_time > 0 else float('inf')
    print(f"\n=> Speedup: {speedup:.2f}x\n")

    if not exact_results:
        print("No results from exact query to compare.")
        return {}

    avg_errors = {}
    # Handle non-grouped (full table aggregate) case
    if not group_by_cols:
        if not approx_results: return {}
        exact_row = exact_results[0]
        approx_row = approx_results[0]
        agg_keys = list(exact_row.keys())

        print(f"--- Results ---")
        header = f"{'Aggregate':<25} | {'Exact Value':>18} | {'Approx. Estimate':>20} | {'Confidence Interval':>40} | {'Actual Error':>15}"
        print(header)
        print("-" * len(header))

        for agg_key in agg_keys:
            exact_val = exact_row.get(agg_key)
            if exact_val is None: continue

            est_key = f"{agg_key}_estimate"
            if est_key not in approx_row: continue

            approx_val = approx_row[est_key]
            error = (abs(approx_val - exact_val) / exact_val) * 100 if exact_val != 0 else 0
            avg_errors[agg_key] = error
            lower_key = f"{agg_key}_lower"
            upper_key = f"{agg_key}_upper"
            ci_str = f"[{approx_row.get(lower_key, 0):,.2f}, {approx_row.get(upper_key, 0):,.2f}]"

            print(f"{agg_key:<25} | {exact_val:18,.2f} | {approx_val:20,.2f} | {ci_str:>40} | {error:14.2f}%")
        print()
        return {'speedup': speedup, 'avg_errors': avg_errors}

    # Handle grouped case
    agg_keys = [k for k in exact_results[0].keys() if k not in group_by_cols]
    exact_map = {tuple(row[k] for k in group_by_cols): row for row in exact_results}

    for agg_key in agg_keys:
        print(f"--- Results for '{agg_key}' ---")
        header = "".join([f"{col:<15} | " for col in
                          group_by_cols]) + f"{'Exact Value':>18} | {'Approx. Estimate':>20} | {'Confidence Interval':>40} | {'Actual Error':>15}"
        print(header)
        print("-" * len(header))

        all_errors = []
        approx_results.sort(key=lambda x: tuple(x.get(k, '') for k in group_by_cols))

        for approx_row in approx_results:
            group_val_tuple = tuple(approx_row.get(k) for k in group_by_cols)
            if any(v is None for v in group_val_tuple): continue

            exact_row = exact_map.get(group_val_tuple)
            if exact_row:
                exact_val = exact_row.get(agg_key)
                if exact_val is None: continue

                est_key = f"{agg_key}_estimate"
                lower_key = f"{agg_key}_lower"
                upper_key = f"{agg_key}_upper"

                if est_key not in approx_row: continue

                approx_val = approx_row[est_key]
                error = (abs(approx_val - exact_val) / exact_val) * 100 if exact_val != 0 else 0
                all_errors.append(error)
                ci_str = f"[{approx_row.get(lower_key, 0):,.2f}, {approx_row.get(upper_key, 0):,.2f}]"

                group_str = "".join([f"{str(v):<15} | " for v in group_val_tuple])
                print(f"{group_str}{exact_val:18,.2f} | {approx_val:20,.2f} | {ci_str:>40} | {error:14.2f}%")

        avg_error = statistics.mean(all_errors) if all_errors else 0
        avg_errors[agg_key] = avg_error
        print(f"\n=> Average Actual Error for '{agg_key}': {avg_error:.2f}%\n")

    return {'speedup': speedup, 'avg_errors': avg_errors}


def plot_results(results_data: List[Dict]):
    """Generates plots for speedup and error vs. sampling ratio."""
    ratios = sorted(list(set([r['ratio'] for r in results_data])))

    query_names = sorted(list(set([r['query_name'] for r in results_data])))

    speedup_data = {name: [] for name in query_names}
    error_data = {name: [] for name in query_names}

    for ratio in ratios:
        for name in query_names:
            # Find the result for this ratio and query name
            res = next((r for r in results_data if r['ratio'] == ratio and r['query_name'] == name), None)
            if res:
                speedup_data[name].append(res.get('speedup', 0))
                # Take the average of all aggregate errors for this query
                avg_err = statistics.mean(res.get('avg_errors', {}).values()) if res.get('avg_errors') else 0
                error_data[name].append(avg_err)
            else:
                speedup_data[name].append(0)
                error_data[name].append(0)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Speedup
    for name in query_names:
        ax1.plot([r * 100 for r in ratios], speedup_data[name], marker='o', linestyle='-', label=name)
    ax1.set_title('Performance Speedup vs. Sampling Ratio', fontsize=16)
    ax1.set_xlabel('Sampling Ratio (%)', fontsize=12)
    ax1.set_ylabel('Speedup (x)', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()

    # Plot 2: Error
    for name in query_names:
        ax2.plot([r * 100 for r in ratios], error_data[name], marker='o', linestyle='-', label=name)
    ax2.set_title('Approximation Error vs. Sampling Ratio', fontsize=16)
    ax2.set_xlabel('Sampling Ratio (%)', fontsize=12)
    ax2.set_ylabel('Average Actual Error (%)', fontsize=12)
    ax2.grid(True, which="both", ls="--")
    ax2.legend()

    fig.tight_layout(pad=3.0)
    plt.show()


def main():
    # ==============================================================================
    # --- USER CONFIGURATION: UPDATE THESE VALUES TO MATCH YOUR DATASET ---
    # ==============================================================================
    # The path is now relative to the root of the project
    DATASET_FILENAME = "data/yellow_tripdata_2015-01.csv"
    TABLE_NAME = "trips"
    COLUMN_TO_STRATIFY_ON = "RateCodeID"
    COLUMN_FOR_HASH_SAMPLING = "payment_type"
    NUMERIC_COLUMN_TO_AGGREGATE = "total_amount"

    # ==============================================================================
    # --- BENCHMARK QUERIES (A mix of simple and grouped queries) ---
    # ==============================================================================
    queries = {
        "Full Table COUNT/AVG": {
            "sql": f'SELECT COUNT(*) AS total_trips, AVG({NUMERIC_COLUMN_TO_AGGREGATE}) as avg_revenue FROM {TABLE_NAME}',
            "group_by_cols": []
        },
        "Full Table SUM": {
            "sql": f'SELECT SUM({NUMERIC_COLUMN_TO_AGGREGATE}) AS total_revenue FROM {TABLE_NAME}',
            "group_by_cols": []
        },
        "Grouped COUNT/AVG (Stratified)": {
            "sql": f'SELECT {COLUMN_TO_STRATIFY_ON}, COUNT(*) AS item_count, AVG({NUMERIC_COLUMN_TO_AGGREGATE}) as avg_value FROM {TABLE_NAME} GROUP BY {COLUMN_TO_STRATIFY_ON}',
            "group_by_cols": [COLUMN_TO_STRATIFY_ON]
        },
        "Grouped SUM (Uniform Fallback)": {
            "sql": f'SELECT {COLUMN_FOR_HASH_SAMPLING}, SUM({NUMERIC_COLUMN_TO_AGGREGATE}) AS total_value FROM {TABLE_NAME} GROUP BY {COLUMN_FOR_HASH_SAMPLING}',
            "group_by_cols": [COLUMN_FOR_HASH_SAMPLING]
        }
    }

    # ==============================================================================

    SAMPLING_RATIOS_TO_TEST = [0.01, 0.02, 0.05, 0.1]
    all_results = []

    engine = ApproxQueryEngine(DB_NAME)

    is_data_loaded = engine.populate_database_from_csv(
        csv_filename=DATASET_FILENAME,
        table_name=TABLE_NAME,
        index_col=COLUMN_TO_STRATIFY_ON
    )

    if not is_data_loaded:
        return

    for ratio in SAMPLING_RATIOS_TO_TEST:
        print("\n" + "=" * 80)
        print(f"BENCHMARKING WITH SAMPLING RATIO: {ratio * 100:.1f}%")
        print("=" * 80 + "\n")

        engine.setup_samples(
            table_name=TABLE_NAME,
            ratio=ratio,
            stratified_col=COLUMN_TO_STRATIFY_ON,
            hash_col=COLUMN_FOR_HASH_SAMPLING
        )

        for query_name, query_info in queries.items():
            benchmark_results = run_benchmark(
                engine=engine,
                query_name=f"{query_name} ({ratio * 100:.1f}% sample)",
                sql_query=query_info["sql"],
                group_by_cols=query_info["group_by_cols"]
            )
            if benchmark_results:
                benchmark_results['query_name'] = query_name
                benchmark_results['ratio'] = ratio
                all_results.append(benchmark_results)

    # After all benchmarks are done, plot the results
    plot_results(all_results)


if __name__ == "__main__":
    main()