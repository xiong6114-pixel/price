from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from case_studies.power.ev_public_charging_case.compare_baselines import summarize_step_metrics
from case_studies.power.ev_public_charging_case.train_rllib import (
    get_paper_response_config,
    run_constant_pricing,
    run_fixed_pricing,
)


def summarize_dir(label: str, directory: Path, q_threshold: float) -> None:
    import csv

    with (directory / "step_metrics.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    summary = summarize_step_metrics(rows, label, q_threshold=q_threshold)
    print(
        f"{label:<16} "
        f"served_kwh={summary['avg_daily_charging_volume_kwh']:9.2f} "
        f"profit={summary['avg_daily_network_profit_cny']:8.2f} "
        f"qvol={summary['queue_volatility']:6.3f} "
        f"viol={summary['queue_violation_count']:6.0f} "
        f"abandoned={summary['total_abandoned']:6.0f}"
    )


def run_one(name: str, cfg: dict) -> None:
    base = Path("D:/price/outputs") / f"paper_response_smoke_{name}"
    base.mkdir(parents=True, exist_ok=True)

    run_fixed_pricing(
        num_episodes=1,
        steps_per_episode=96,
        seed=2026,
        env_config=cfg,
        output_dir=str(base / "FP_eval"),
    )
    run_constant_pricing(
        price=0.15,
        num_episodes=1,
        steps_per_episode=96,
        seed=2026,
        env_config=cfg,
        output_dir=str(base / "CONST_LOW_eval"),
        algo="CONST-LOW",
    )
    run_constant_pricing(
        price=0.40,
        num_episodes=1,
        steps_per_episode=96,
        seed=2026,
        env_config=cfg,
        output_dir=str(base / "CONST_HIGH_eval"),
        algo="CONST-HIGH",
    )

    print(f"\n=== {name} ===")
    q_threshold = float(cfg.get("q_threshold", 4.0))
    summarize_dir("FP", base / "FP_eval", q_threshold)
    summarize_dir("CONST-LOW", base / "CONST_LOW_eval", q_threshold)
    summarize_dir("CONST-HIGH", base / "CONST_HIGH_eval", q_threshold)


def main() -> None:
    candidates = [
        ("E_q4_max12_gc120_wp035_soft", get_paper_response_config(
            q_threshold=4.0,
            max_queue_size=12,
            generalized_cost_threshold=120.0,
            omega_price=0.35,
            choice_lmp_weight=0.0,
            charge_lmp_weight=0.0,
        )),
        ("F_q5_max12_gc120_wp035_soft", get_paper_response_config(
            q_threshold=5.0,
            max_queue_size=12,
            generalized_cost_threshold=120.0,
            omega_price=0.35,
            choice_lmp_weight=0.0,
            charge_lmp_weight=0.0,
        )),
        ("G_q4_max12_gc100_wp025_soft", get_paper_response_config(
            q_threshold=4.0,
            max_queue_size=12,
            generalized_cost_threshold=100.0,
            omega_price=0.25,
            choice_lmp_weight=0.0,
            charge_lmp_weight=0.0,
        )),
    ]
    for name, cfg in candidates:
        run_one(name, cfg)


if __name__ == "__main__":
    main()
