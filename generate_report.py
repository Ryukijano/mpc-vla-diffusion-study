#!/usr/bin/env python3
"""Report generator for the MPC vs VLA vs Diffusion comparison study.

Reads result CSV files from the ``results/`` directory and generates:

  * **Master comparison table** (CSV + formatted console output, optional LaTeX).
  * **Bar charts** comparing all methods on success rate, latency, and mode
    coverage -> ``figures/``.
  * **Latency-performance Pareto plot** -> ``figures/``.
  * **Ablation bar chart** (from EXP-001 outputs) -> ``figures/``.

Usage::

    conda run -n mpc_vla python generate_report.py
    conda run -n mpc_vla python generate_report.py --results-dir results --output-dir report --format both
    conda run -n mpc_vla python generate_report.py --format latex
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
STUDY_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STUDY_ROOT)


# ===========================================================================
# CSV reading helpers
# ===========================================================================
def _read_csv(path: str) -> List[Dict[str, str]]:
    """Read a CSV file and return a list of row dicts."""
    rows = []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"  [WARNING] File not found: {path}")
    except Exception as exc:
        print(f"  [WARNING] Error reading {path}: {exc}")
    return rows


def _safe_float(val, default=float("nan")):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def find_csv_files(results_dir: str) -> Dict[str, str]:
    """Find relevant CSV files in the results directory tree."""
    found = {}

    # Master comparison (per-seed)
    for candidate in [
        os.path.join(results_dir, "tables", "master_comparison.csv"),
        os.path.join(results_dir, "master_comparison.csv"),
    ]:
        if os.path.isfile(candidate):
            found["master"] = candidate
            break

    # Aggregated comparison
    for candidate in [
        os.path.join(results_dir, "tables", "aggregated_comparison.csv"),
        os.path.join(results_dir, "aggregated_comparison.csv"),
    ]:
        if os.path.isfile(candidate):
            found["aggregated"] = candidate
            break

    # Ablation results
    for candidate in [
        os.path.join(results_dir, "experiments", "EXP-001-mechanism-ablation",
                      "outputs", "ablation_aggregated.csv"),
        os.path.join(results_dir, "ablation_aggregated.csv"),
        os.path.join(STUDY_ROOT, "experiments", "EXP-001-mechanism-ablation",
                      "outputs", "ablation_aggregated.csv"),
    ]:
        if os.path.isfile(candidate):
            found["ablation"] = candidate
            break

    # Also search for ablation in a separate ablation dir
    ablation_dir = os.path.join(STUDY_ROOT, "experiments",
                                 "EXP-001-mechanism-ablation", "outputs")
    if "ablation" not in found and os.path.isdir(ablation_dir):
        candidate = os.path.join(ablation_dir, "ablation_aggregated.csv")
        if os.path.isfile(candidate):
            found["ablation"] = candidate

    return found


# ===========================================================================
# Table generation
# ===========================================================================
def generate_master_table(aggregated_rows: List[Dict], output_dir: str,
                           fmt: str = "csv") -> str:
    """Generate the master comparison table.

    Returns the path to the saved CSV (or LaTeX) file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Console output ---
    print("\n" + "=" * 100)
    print("MASTER COMPARISON TABLE (aggregated over seeds)")
    print("=" * 100)

    # Group by benchmark
    by_bench: Dict[str, List[Dict]] = {}
    for row in aggregated_rows:
        bn = row.get("benchmark", "unknown")
        by_bench.setdefault(bn, []).append(row)

    for bn, rows in sorted(by_bench.items()):
        print(f"\n  Benchmark: {bn}")
        header = (f"  {'Controller':<32} {'Success':>8} {'±std':>7} "
                  f"{'PathLen':>9} {'CollRate':>9} {'Lat(ms)':>9} {'ModeCov':>8}")
        print(header)
        print("  " + "-" * 86)
        for r in sorted(rows, key=lambda x: x.get("controller", "")):
            ctrl = r.get("controller", "?")
            sr = _safe_float(r.get("success_rate_mean"))
            sr_std = _safe_float(r.get("success_rate_std"))
            pl = _safe_float(r.get("path_length_mean"))
            cr = _safe_float(r.get("collision_rate_mean"))
            lat = _safe_float(r.get("latency_ms_mean"))
            mc = _safe_float(r.get("mode_coverage_mean"))
            print(f"  {ctrl:<32} {sr:>8.3f} {sr_std:>7.3f} "
                  f"{pl:>9.3f} {cr:>9.3f} {lat:>9.2f} {mc:>8.3f}")
    print("\n" + "=" * 100)

    # --- CSV output ---
    csv_path = os.path.join(output_dir, "master_comparison_table.csv")
    if aggregated_rows:
        fieldnames = list(aggregated_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregated_rows)
        print(f"  Master table CSV saved to {csv_path}")

    # --- LaTeX output ---
    if fmt in ("latex", "both"):
        latex_path = os.path.join(output_dir, "master_comparison_table.tex")
        _write_latex_table(aggregated_rows, latex_path)
        print(f"  Master table LaTeX saved to {latex_path}")

    return csv_path


def _write_latex_table(rows: List[Dict], path: str):
    """Write a LaTeX table from aggregated rows."""
    if not rows:
        return

    cols = ["benchmark", "controller", "success_rate_mean", "success_rate_std",
            "path_length_mean", "collision_rate_mean", "latency_ms_mean",
            "mode_coverage_mean"]
    col_labels = ["Benchmark", "Controller", "Success", "±std", "PathLen",
                  "CollRate", "Lat(ms)", "ModeCov"]

    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Master Comparison: MPC vs VLA vs Diffusion (mean over seeds)}\n")
        f.write("\\label{tab:master_comparison}\n")
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(col_labels) + " \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                fv = _safe_float(v, None)
                if fv is not None:
                    vals.append(f"{fv:.3f}")
                else:
                    vals.append(str(v))
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


# ===========================================================================
# Plot generation
# ===========================================================================
def generate_bar_charts(aggregated_rows: List[Dict], figures_dir: str):
    """Generate bar charts comparing all methods."""
    if not aggregated_rows:
        print("  [WARNING] No data for bar charts")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available -- skipping bar charts")
        return

    os.makedirs(figures_dir, exist_ok=True)

    # Group by benchmark
    by_bench: Dict[str, List[Dict]] = {}
    for row in aggregated_rows:
        bn = row.get("benchmark", "unknown")
        by_bench.setdefault(bn, []).append(row)

    bench_names = sorted(by_bench.keys())
    if not bench_names:
        print("  [WARNING] No benchmarks found for bar charts")
        return

    # Collect all controller names (union)
    all_controllers = []
    for bn in bench_names:
        for r in by_bench[bn]:
            ctrl = r.get("controller", "?")
            if ctrl not in all_controllers:
                all_controllers.append(ctrl)

    n_ctrl = len(all_controllers)
    n_bench = len(bench_names)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0",
              "#00BCD4", "#795548", "#CDDC39"]

    # --- Success Rate ---
    fig, ax = plt.subplots(figsize=(max(10, n_ctrl * 1.5), 6))
    x = np.arange(n_ctrl)
    width = 0.8 / max(n_bench, 1)
    for i, bn in enumerate(bench_names):
        values = []
        for ctrl in all_controllers:
            match = [r for r in by_bench[bn] if r.get("controller") == ctrl]
            if match:
                values.append(_safe_float(match[0].get("success_rate_mean")))
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Controller", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Success Rate Comparison (mean over seeds)", fontsize=14)
    ax.set_xticks(x + width * (n_bench - 1) / 2)
    ax.set_xticklabels([c[:22] for c in all_controllers], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "comparison_success_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Success rate chart saved to {path}")

    # --- Latency ---
    fig, ax = plt.subplots(figsize=(max(10, n_ctrl * 1.5), 6))
    for i, bn in enumerate(bench_names):
        values = []
        for ctrl in all_controllers:
            match = [r for r in by_bench[bn] if r.get("controller") == ctrl]
            if match:
                lat = _safe_float(match[0].get("latency_ms_mean"))
                values.append(lat if not np.isnan(lat) else 0.0)
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Controller", fontsize=12)
    ax.set_ylabel("Inference Latency (ms)", fontsize=12)
    ax.set_title("Inference Latency Comparison (mean over seeds)", fontsize=14)
    ax.set_xticks(x + width * (n_bench - 1) / 2)
    ax.set_xticklabels([c[:22] for c in all_controllers], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "comparison_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Latency chart saved to {path}")

    # --- Mode Coverage ---
    fig, ax = plt.subplots(figsize=(max(10, n_ctrl * 1.5), 6))
    for i, bn in enumerate(bench_names):
        values = []
        for ctrl in all_controllers:
            match = [r for r in by_bench[bn] if r.get("controller") == ctrl]
            if match:
                values.append(_safe_float(match[0].get("mode_coverage_mean")))
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Controller", fontsize=12)
    ax.set_ylabel("Mode Coverage", fontsize=12)
    ax.set_title("Mode Coverage Comparison (mean over seeds)", fontsize=14)
    ax.set_xticks(x + width * (n_bench - 1) / 2)
    ax.set_xticklabels([c[:22] for c in all_controllers], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "comparison_mode_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Mode coverage chart saved to {path}")

    # --- Collision Rate ---
    fig, ax = plt.subplots(figsize=(max(10, n_ctrl * 1.5), 6))
    for i, bn in enumerate(bench_names):
        values = []
        for ctrl in all_controllers:
            match = [r for r in by_bench[bn] if r.get("controller") == ctrl]
            if match:
                values.append(_safe_float(match[0].get("collision_rate_mean")))
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Controller", fontsize=12)
    ax.set_ylabel("Collision Rate", fontsize=12)
    ax.set_title("Collision Rate Comparison (mean over seeds)", fontsize=14)
    ax.set_xticks(x + width * (n_bench - 1) / 2)
    ax.set_xticklabels([c[:22] for c in all_controllers], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "comparison_collision_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Collision rate chart saved to {path}")


def generate_pareto_plot(aggregated_rows: List[Dict], figures_dir: str):
    """Generate a latency vs. success rate Pareto plot."""
    if not aggregated_rows:
        print("  [WARNING] No data for Pareto plot")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available -- skipping Pareto plot")
        return

    os.makedirs(figures_dir, exist_ok=True)

    # Group by benchmark
    by_bench: Dict[str, List[Dict]] = {}
    for row in aggregated_rows:
        bn = row.get("benchmark", "unknown")
        by_bench.setdefault(bn, []).append(row)

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0",
              "#00BCD4", "#795548", "#CDDC39"]

    fig, ax = plt.subplots(figsize=(10, 7))

    for bi, bn in enumerate(sorted(by_bench.keys())):
        rows = by_bench[bn]
        for ri, r in enumerate(rows):
            sr = _safe_float(r.get("success_rate_mean"))
            lat = _safe_float(r.get("latency_ms_mean"))
            ctrl = r.get("controller", "?")
            if np.isnan(sr) or np.isnan(lat):
                continue
            mi = (bi * len(rows) + ri) % len(markers)
            ci = bi % len(colors)
            ax.scatter(lat, sr, marker=markers[mi], color=colors[ci],
                       s=120, alpha=0.8, edgecolors="black", linewidths=0.5,
                       label=f"{bn}/{ctrl}" if bi == 0 else None)
            ax.annotate(ctrl[:15], (lat, sr), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.7)

    # Pareto frontier (per benchmark, highlight non-dominated points)
    for bn in sorted(by_bench.keys()):
        rows = by_bench[bn]
        points = []
        for r in rows:
            sr = _safe_float(r.get("success_rate_mean"))
            lat = _safe_float(r.get("latency_ms_mean"))
            if not (np.isnan(sr) or np.isnan(lat)):
                points.append((lat, sr, r.get("controller", "?")))
        if len(points) < 2:
            continue
        # Sort by latency; find Pareto-optimal (high success, low latency)
        points.sort(key=lambda p: p[0])
        pareto = []
        best_sr = -1
        for lat, sr, ctrl in points:
            if sr > best_sr:
                pareto.append((lat, sr))
                best_sr = sr
        if len(pareto) >= 2:
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.plot(px, py, "k--", alpha=0.3, linewidth=1.5)

    ax.set_xlabel("Inference Latency (ms)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Latency-Performance Pareto Plot", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    # Add legend for benchmarks
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=markers[i % len(markers)], color=colors[i % len(colors)],
               label=bn, markersize=10, linestyle="None")
        for i, bn in enumerate(sorted(by_bench.keys()))
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    fig.tight_layout()
    path = os.path.join(figures_dir, "pareto_latency_vs_success.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Pareto plot saved to {path}")


def generate_ablation_chart(ablation_rows: List[Dict], figures_dir: str):
    """Generate ablation bar chart from EXP-001 results."""
    if not ablation_rows:
        print("  [SKIP] No ablation data found -- skipping ablation chart")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available -- skipping ablation chart")
        return

    os.makedirs(figures_dir, exist_ok=True)

    by_bench: Dict[str, List[Dict]] = {}
    for row in ablation_rows:
        bn = row.get("benchmark", "unknown")
        by_bench.setdefault(bn, []).append(row)

    bench_names = sorted(by_bench.keys())
    all_variants = []
    for bn in bench_names:
        for r in by_bench[bn]:
            v = r.get("variant", "?")
            if v not in all_variants:
                all_variants.append(v)

    n_var = len(all_variants)
    n_bench = len(bench_names)
    if n_var == 0:
        return

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]
    x = np.arange(n_var)
    width = 0.8 / max(n_bench, 1)

    # --- Success Rate ---
    fig, ax = plt.subplots(figsize=(max(10, n_var * 1.8), 6))
    for i, bn in enumerate(bench_names):
        values = []
        for v in all_variants:
            match = [r for r in by_bench[bn] if r.get("variant") == v]
            if match:
                values.append(_safe_float(match[0].get("success_rate_mean")))
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("EXP-001: GCP Ablation -- Success Rate", fontsize=14)
    ax.set_xticks(x + width * (n_bench - 1) / 2)
    ax.set_xticklabels([v[:22] for v in all_variants], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "ablation_success_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Ablation success chart saved to {path}")

    # --- Latency ---
    fig, ax = plt.subplots(figsize=(max(10, n_var * 1.8), 6))
    for i, bn in enumerate(bench_names):
        values = []
        for v in all_variants:
            match = [r for r in by_bench[bn] if r.get("variant") == v]
            if match:
                lat = _safe_float(match[0].get("latency_ms_mean"))
                values.append(lat if not np.isnan(lat) else 0.0)
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Inference Latency (ms)", fontsize=12)
    ax.set_title("EXP-001: GCP Ablation -- Inference Latency", fontsize=14)
    ax.set_xticks(x + width * (n_bench - 1) / 2)
    ax.set_xticklabels([v[:22] for v in all_variants], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "ablation_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Ablation latency chart saved to {path}")

    # --- Mode Coverage ---
    fig, ax = plt.subplots(figsize=(max(10, n_var * 1.8), 6))
    for i, bn in enumerate(bench_names):
        values = []
        for v in all_variants:
            match = [r for r in by_bench[bn] if r.get("variant") == v]
            if match:
                values.append(_safe_float(match[0].get("mode_coverage_mean")))
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Mode Coverage", fontsize=12)
    ax.set_title("EXP-001: GCP Ablation -- Mode Coverage", fontsize=14)
    ax.set_xticks(x + width * (n_bench - 1) / 2)
    ax.set_xticklabels([v[:22] for v in all_variants], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "ablation_mode_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Ablation mode coverage chart saved to {path}")


# ===========================================================================
# Main report generation
# ===========================================================================
def generate_report(results_dir: str, output_dir: str, fmt: str = "csv"):
    print("\n" + "=" * 90)
    print("Report Generator: MPC vs VLA vs Diffusion Study")
    print("=" * 90)
    print(f"  Results dir:  {results_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Format:       {fmt}")

    # --- Find CSV files ---
    csv_files = find_csv_files(results_dir)
    print(f"  Found CSVs:   {list(csv_files.keys())}")

    if not csv_files:
        print("  [WARNING] No result CSV files found. Run run_experiments.py first.")
        print(f"  Expected location: {os.path.join(results_dir, 'tables', 'aggregated_comparison.csv')}")
        return

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # --- Read aggregated data ---
    aggregated_rows = []
    if "aggregated" in csv_files:
        aggregated_rows = _read_csv(csv_files["aggregated"])
        print(f"  Read {len(aggregated_rows)} aggregated rows")
    elif "master" in csv_files:
        # Aggregate master rows ourselves
        master_rows = _read_csv(csv_files["master"])
        print(f"  Read {len(master_rows)} master rows (will aggregate)")
        aggregated_rows = _aggregate_master_rows(master_rows)

    # --- Generate master table ---
    if aggregated_rows:
        generate_master_table(aggregated_rows, output_dir, fmt=fmt)
    else:
        print("  [WARNING] No aggregated data -- skipping master table")

    # --- Generate bar charts ---
    if aggregated_rows:
        print("\n  Generating comparison bar charts...")
        generate_bar_charts(aggregated_rows, figures_dir)

    # --- Generate Pareto plot ---
    if aggregated_rows:
        print("\n  Generating Pareto plot...")
        generate_pareto_plot(aggregated_rows, figures_dir)

    # --- Generate ablation chart ---
    ablation_rows = []
    if "ablation" in csv_files:
        ablation_rows = _read_csv(csv_files["ablation"])
        print(f"  Read {len(ablation_rows)} ablation rows")

    if ablation_rows:
        print("\n  Generating ablation charts...")
        generate_ablation_chart(ablation_rows, figures_dir)
    else:
        print("  [INFO] No ablation data found -- run run_ablation.py for EXP-001 charts")

    # --- Summary ---
    print("\n" + "=" * 90)
    print("Report generation complete!")
    print(f"  Tables saved to:   {output_dir}/")
    print(f"  Figures saved to:  {figures_dir}/")
    print("=" * 90)


def _aggregate_master_rows(master_rows: List[Dict]) -> List[Dict]:
    """Aggregate per-seed master rows into mean/std per (benchmark, controller)."""
    groups: Dict[tuple, List[Dict]] = {}
    for r in master_rows:
        key = (r.get("benchmark", ""), r.get("controller", ""))
        groups.setdefault(key, []).append(r)

    aggregated = []
    for (bn, ctrl), rows in groups.items():
        sr = [_safe_float(r.get("success_rate")) for r in rows]
        pl = [_safe_float(r.get("path_length")) for r in rows]
        cr = [_safe_float(r.get("collision_rate")) for r in rows]
        lat = [_safe_float(r.get("latency_ms")) for r in rows]
        mc = [_safe_float(r.get("mode_coverage")) for r in rows]
        n = len(rows)
        aggregated.append({
            "benchmark": bn,
            "controller": ctrl,
            "success_rate_mean": float(np.mean(sr)) if sr else 0.0,
            "success_rate_std": float(np.std(sr)) if sr else 0.0,
            "path_length_mean": float(np.mean(pl)) if pl else 0.0,
            "collision_rate_mean": float(np.mean(cr)) if cr else 0.0,
            "latency_ms_mean": float(np.mean(lat)) if lat else 0.0,
            "mode_coverage_mean": float(np.mean(mc)) if mc else 0.0,
            "n_seeds": n,
            "n_episodes": rows[0].get("n_episodes", "") if rows else "",
        })
    return aggregated


# ===========================================================================
# Argument parsing
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Report generator for the MPC vs VLA vs Diffusion study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report from default results/ dir
  conda run -n mpc_vla python generate_report.py

  # Custom dirs, both CSV and LaTeX
  conda run -n mpc_vla python generate_report.py --results-dir results --output-dir report --format both

  # LaTeX only
  conda run -n mpc_vla python generate_report.py --format latex
        """,
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing result CSVs (default: results/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for report (default: results/report/)",
    )
    parser.add_argument(
        "--format", type=str, default="csv",
        choices=["csv", "latex", "both"],
        help="Output format for tables (default: csv)",
    )

    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(STUDY_ROOT, "results")
    output_dir = args.output_dir or os.path.join(STUDY_ROOT, "results", "report")

    generate_report(results_dir, output_dir, fmt=args.format)


if __name__ == "__main__":
    main()
