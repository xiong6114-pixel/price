"""Compare baseline result directories and export a unified summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional


def read_csv_if_exists(path: Path) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[WARN] Empty file: {path}")
        return None
    return rows


def infer_algo(rows: List[Dict[str, str]], fallback: str) -> str:
    for row in rows:
        algo = row.get("algo", "").strip()
        if algo:
            return algo
    return fallback


def _to_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else math.nan


def summarize_step_metrics(
    step_rows: List[Dict[str, str]],
    algo: str,
    q_threshold: float = 4.0,
) -> Dict[str, float]:
    required_cols = [
        "served_kwh",
        "revenue",
        "grid_cost",
        "profit",
        "queue_len",
        "utilization",
        "abandoned",
        "congestion_penalty",
        "reward",
        "price",
    ]
    missing = [c for c in required_cols if c not in step_rows[0]]
    if missing:
        raise ValueError(f"{algo}: step_metrics.csv missing columns: {missing}")

    episodes = len({row["episode"] for row in step_rows if "episode" in row})
    steps = len({row["step"] for row in step_rows if "step" in row})
    stations = len({row["station"] for row in step_rows if "station" in row})

    served_kwh = [_to_float(row, "served_kwh") for row in step_rows]
    revenue = [_to_float(row, "revenue") for row in step_rows]
    grid_cost = [_to_float(row, "grid_cost") for row in step_rows]
    profit = [_to_float(row, "profit") for row in step_rows]
    reward = [_to_float(row, "reward") for row in step_rows]
    abandoned = [_to_float(row, "abandoned") for row in step_rows]
    queue_len = [_to_float(row, "queue_len") for row in step_rows]
    utilization = [_to_float(row, "utilization") for row in step_rows]
    congestion_penalty = [_to_float(row, "congestion_penalty") for row in step_rows]
    price = [_to_float(row, "price") for row in step_rows]
    queue_violation_flags = [1.0 if q > q_threshold else 0.0 for q in queue_len]
    queue_excess = [max(0.0, q - q_threshold) for q in queue_len]
    network_queues = defaultdict(list)
    for row in step_rows:
        episode = row.get("episode")
        step = row.get("step")
        q = _to_float(row, "queue_len")
        network_queues[(episode, step)].append(q)
    network_max_queue = [max(qs) for qs in network_queues.values() if qs]
    network_violation_flags = [1.0 if q > q_threshold else 0.0 for q in network_max_queue]
    network_queue_excess = [max(0.0, q - q_threshold) for q in network_max_queue]
    abandoned_soc = [
        _to_float(row, "abandoned_soc") if "abandoned_soc" in row else 0.0
        for row in step_rows
    ]
    abandoned_cost = [
        _to_float(row, "abandoned_cost") if "abandoned_cost" in row else 0.0
        for row in step_rows
    ]
    abandoned_full = [
        _to_float(row, "abandoned_full") if "abandoned_full" in row else 0.0
        for row in step_rows
    ]
    abandoned_timeout = [
        _to_float(row, "abandoned_timeout") if "abandoned_timeout" in row else 0.0
        for row in step_rows
    ]
    daily_divisor = max(float(episodes), 1.0)

    return {
        "algo": algo,
        "episodes": float(episodes),
        "steps_per_episode": float(steps),
        "stations": float(stations),
        "avg_daily_charging_volume_kwh": sum(served_kwh) / daily_divisor,
        "avg_daily_total_revenue_cny": sum(revenue) / daily_divisor,
        "avg_daily_electricity_cost_cny": sum(grid_cost) / daily_divisor,
        "avg_daily_network_profit_cny": sum(profit) / daily_divisor,
        "total_served_kwh": sum(served_kwh),
        "total_revenue": sum(revenue),
        "total_grid_cost": sum(grid_cost),
        "total_profit": sum(profit),
        "total_reward_from_steps": sum(reward),
        "total_abandoned": sum(abandoned),
        "total_abandoned_soc": sum(abandoned_soc),
        "total_abandoned_cost": sum(abandoned_cost),
        "total_abandoned_full": sum(abandoned_full),
        "total_abandoned_timeout": sum(abandoned_timeout),
        "avg_abandoned_per_step": mean(abandoned),
        "avg_queue_len": mean(queue_len),
        "max_queue_len": max(queue_len),
        "queue_volatility": pstdev(queue_len) if len(queue_len) > 1 else 0.0,
        "network_queue_volatility": pstdev(network_max_queue) if len(network_max_queue) > 1 else 0.0,
        "queue_violation_count": sum(queue_violation_flags),
        "station_queue_violation_count": sum(queue_violation_flags),
        "network_queue_violation_count": sum(network_violation_flags),
        "queue_excess_total": sum(queue_excess),
        "queue_excess_mean": mean(queue_excess),
        "network_queue_excess_total": sum(network_queue_excess),
        "network_queue_excess_mean": mean(network_queue_excess) if network_queue_excess else 0.0,
        "avg_utilization": mean(utilization),
        "max_utilization": max(utilization),
        "total_congestion_penalty": sum(congestion_penalty),
        "paper_violation_count_proxy": sum(queue_violation_flags),
        "paper_violation_plus_abandoned_proxy": sum(queue_violation_flags) + sum(abandoned),
        "avg_price": mean(price),
        "min_price": min(price),
        "max_price": max(price),
    }


def summarize_episode_metrics(episode_rows: Optional[List[Dict[str, str]]], algo: str) -> Dict[str, float]:
    if episode_rows is None:
        return {
            "total_reward_from_episodes": math.nan,
            "avg_episode_reward": math.nan,
            "std_episode_reward": math.nan,
        }

    if "total_reward" not in episode_rows[0]:
        print(f"[WARN] {algo}: episode_metrics.csv has no total_reward column.")
        return {
            "total_reward_from_episodes": math.nan,
            "avg_episode_reward": math.nan,
            "std_episode_reward": math.nan,
        }

    totals = [_to_float(row, "total_reward") for row in episode_rows]
    return {
        "total_reward_from_episodes": sum(totals),
        "avg_episode_reward": mean(totals),
        "std_episode_reward": pstdev(totals),
    }


def load_one_baseline(label: str, directory: Path, q_threshold: float = 4.0) -> Dict[str, float]:
    step_path = directory / "step_metrics.csv"
    episode_path = directory / "episode_metrics.csv"

    step_rows = read_csv_if_exists(step_path)
    if step_rows is None:
        raise FileNotFoundError(f"Cannot compare {label}; missing valid {step_path}")

    episode_rows = read_csv_if_exists(episode_path)
    algo = infer_algo(step_rows, fallback=label)

    step_summary = summarize_step_metrics(step_rows, algo, q_threshold=q_threshold)
    episode_summary = summarize_episode_metrics(episode_rows, algo)
    return {**step_summary, **episode_summary}


def pct_gain(value: float, base: float) -> float:
    if not math.isfinite(base) or abs(base) < 1e-9:
        return math.nan
    return (value - base) / abs(base) * 100.0


def add_relative_metrics(rows: List[Dict[str, float]], fp_algo_name: str = "FP") -> List[Dict[str, float]]:
    fp_row = next((row for row in rows if row.get("algo") == fp_algo_name), None)
    if fp_row is None:
        print("[WARN] No FP row found. Relative improvement columns will be NaN.")
        for row in rows:
            row["profit_gain_vs_fp_pct"] = math.nan
            row["reward_gain_vs_fp_pct"] = math.nan
        return rows

    fp_profit = float(fp_row["total_profit"])
    fp_reward = float(fp_row["total_reward_from_episodes"])
    for row in rows:
        row["profit_gain_vs_fp_pct"] = pct_gain(float(row["total_profit"]), fp_profit)
        row["reward_gain_vs_fp_pct"] = pct_gain(float(row["total_reward_from_episodes"]), fp_reward)
    return rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def print_table(rows: List[Dict[str, float]]) -> None:
    display_cols = [
        "algo",
        "episodes",
        "steps_per_episode",
        "stations",
        "total_profit",
        "total_reward_from_episodes",
        "total_served_kwh",
        "avg_daily_charging_volume_kwh",
        "avg_daily_total_revenue_cny",
        "avg_daily_electricity_cost_cny",
        "avg_daily_network_profit_cny",
        "total_abandoned",
        "queue_volatility",
        "network_queue_volatility",
        "queue_violation_count",
        "network_queue_violation_count",
        "avg_queue_len",
        "avg_utilization",
        "avg_price",
        "profit_gain_vs_fp_pct",
        "reward_gain_vs_fp_pct",
    ]
    available_cols = [col for col in display_cols if any(col in row for row in rows)]
    widths = {
        col: max(len(col), *(len(format_value(row.get(col, ""))) for row in rows))
        for col in available_cols
    }

    header = " ".join(col.ljust(widths[col]) for col in available_cols)
    print(header)
    for row in rows:
        print(" ".join(format_value(row.get(col, "")).ljust(widths[col]) for col in available_cols))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        required=True,
        help="Add one baseline result directory, e.g. --baseline FP outputs/fp --baseline SimpleAC outputs",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/baseline_comparison.csv",
        help="Output comparison CSV path.",
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=4.0,
        help="Queue threshold used to compute paper-style violation counts.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, float]] = []
    for label, directory in args.baseline:
        row = load_one_baseline(label, Path(directory), q_threshold=args.q_threshold)
        if not row.get("algo"):
            row["algo"] = label
        rows.append(row)

    rows = add_relative_metrics(rows, fp_algo_name="FP")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)

    print("\n=== Baseline comparison ===")
    print_table(rows)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
