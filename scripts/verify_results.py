#!/usr/bin/env python3
"""Quick verification of the CPU low-latency Pareto sweep results."""
import csv
import json
from collections import Counter
from pathlib import Path

OUT = Path("results/exp004_cpu_low_latency")


def main():
    assert (OUT / "pareto_data.csv").exists()
    assert (OUT / "latency_table.csv").exists()
    assert (OUT / "pareto_frontier.png").exists()

    with open(OUT / "pareto_data.csv") as f:
        rows = list(csv.DictReader(f))

    n_rows = len(rows)
    benchmarks = sorted(set(r["benchmark"] for r in rows))
    cond_by_bench = Counter(r["benchmark"] for r in rows)

    assert n_rows >= 15, f"Only {n_rows} rows, expected >= 15"
    assert len(benchmarks) >= 2, f"Only {len(benchmarks)} benchmarks, expected >= 2"

    print(f"pareto_data.csv rows: {n_rows}")
    print(f"benchmarks: {benchmarks} (n={len(benchmarks)})")
    print(f"conditions by benchmark: {dict(cond_by_bench)}")

    for bn in benchmarks:
        pts = [r for r in rows if r["benchmark"] == bn and r["is_pareto_optimal_low_latency"] == "True"]
        print(f"\n{bn.upper()} low-latency (<100 ms) Pareto-optimal controllers:")
        for r in sorted(pts, key=lambda x: float(x["latency_p50_ms"])):
            print(
                f"  {r['condition']:30s} p50={float(r['latency_p50_ms']):6.2f}ms  "
                f"success={float(r['success_rate_mean'])*100:5.1f}%"
            )

    with open(OUT / "metrics_summary.json") as f:
        summary = json.load(f)
    print(f"\nn_conditions: {summary['n_conditions']}, seeds: {summary['seeds']}, episodes: {summary['episodes']}")
    print("\nAll required output files present and verified.")


if __name__ == "__main__":
    main()
