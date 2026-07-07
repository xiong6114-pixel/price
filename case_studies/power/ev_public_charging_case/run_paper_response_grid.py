from pathlib import Path
import sys
import argparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from case_studies.power.ev_public_charging_case.compare_baselines import (
    add_relative_metrics,
    load_one_baseline,
    write_csv,
)
from case_studies.power.ev_public_charging_case.train_rllib import (
    evaluate_independent_transa3c,
    evaluate_mappo_mlp,
    evaluate_ma_transa3c,
    iter_paper_response_grid,
    run_fixed_pricing,
    train_independent_transa3c,
    train_mappo_mlp,
    train_ma_transa3c,
)


def _write_comparison(base: Path, q_threshold: float, ma_eval_dir: str, ma_label: str) -> None:
    rows = [
        load_one_baseline("FP", base / "FP_eval", q_threshold=q_threshold),
        load_one_baseline("I-TransA3C", base / "I_TransA3C_eval", q_threshold=q_threshold),
        load_one_baseline("MAPPO", base / "MAPPO_MLP_eval", q_threshold=q_threshold),
        load_one_baseline(ma_label, base / ma_eval_dir, q_threshold=q_threshold),
    ]
    write_csv(base / "paper_table2_reproduction.csv", add_relative_metrics(rows, fp_algo_name="FP"))


def run_one(
    name,
    cfg,
    train_episodes=30,
    eval_episodes=None,
    validation_episodes=5,
    real_order_data_path=None,
    real_order_sheet_name="Sheet2",
    real_order_arrival_scale=1.0,
    real_order_date_filter="dominant_month",
    real_order_use_lmp=False,
    real_order_train_days=20,
    real_order_validation_days=5,
    real_order_test_days=6,
):
    cfg = dict(cfg)
    base_name = name if str(name).startswith("paper_response_") else f"paper_response_{name}"
    resolved_eval_episodes = eval_episodes
    if real_order_data_path:
        cfg.update({
            "real_order_data_path": real_order_data_path,
            "real_order_sheet_name": real_order_sheet_name,
            "real_order_arrival_scale": real_order_arrival_scale,
            "real_order_date_filter": real_order_date_filter,
            "real_order_use_lmp": real_order_use_lmp,
            "real_order_train_days": real_order_train_days,
            "real_order_validation_days": real_order_validation_days,
            "real_order_test_days": real_order_test_days,
        })
        if resolved_eval_episodes is None:
            resolved_eval_episodes = max(int(real_order_test_days), 1)
        scale_tag = "" if abs(float(real_order_arrival_scale) - 1.0) < 1e-9 else f"_scale{real_order_arrival_scale:g}"
        split_tag = f"_split{int(real_order_train_days)}-{int(real_order_validation_days)}-{int(real_order_test_days)}"
        base_name = f"{base_name}_realorders{scale_tag}{split_tag}"
    if resolved_eval_episodes is None:
        resolved_eval_episodes = 1
    base = Path("D:/price/outputs") / base_name
    base.mkdir(parents=True, exist_ok=True)

    run_fixed_pricing(
        num_episodes=resolved_eval_episodes,
        steps_per_episode=96,
        seed=2026,
        env_config=cfg,
        output_dir=str(base / "FP_eval"),
    )

    _, i_policies, _ = train_independent_transa3c(
        num_episodes=train_episodes,
        steps_per_episode=96,
        seed=42,
        gamma=0.99,
        seq_len=8,
        validation_num_episodes=validation_episodes,
        env_config=cfg,
        output_dir=str(base / "I_TransA3C_train"),
        algo="I-TransA3C",
    )
    evaluate_independent_transa3c(
        i_policies,
        num_episodes=resolved_eval_episodes,
        steps_per_episode=96,
        seed=2026,
        seq_len=8,
        env_config=cfg,
        output_dir=str(base / "I_TransA3C_eval"),
        algo="I-TransA3C",
    )

    _, mappo_policy, _ = train_mappo_mlp(
        num_episodes=train_episodes,
        steps_per_episode=96,
        seed=42,
        gamma=0.99,
        use_ma_station_obs=True,
        validation_num_episodes=validation_episodes,
        env_config=cfg,
        output_dir=str(base / "MAPPO_MLP_train"),
        algo="MAPPO-MLP",
    )
    evaluate_mappo_mlp(
        mappo_policy,
        num_episodes=resolved_eval_episodes,
        steps_per_episode=96,
        seed=2026,
        use_ma_station_obs=True,
        env_config=cfg,
        output_dir=str(base / "MAPPO_MLP_eval"),
        algo="MAPPO-MLP",
    )

    ma_train_kwargs = {
        "num_episodes": train_episodes,
        "steps_per_episode": 96,
        "seed": 42,
        "gamma": 0.99,
        "k_neighbors": 4,
        "use_ma_station_obs": True,
        "use_lagged_rank_loss": False,
        "lambda_rank": 0.20,
        "rank_margin": 0.02,
        "rank_eps": 0.10,
        "validation_num_episodes": validation_episodes,
        "env_config": cfg,
        "output_dir": str(base / "MA_v3_1b_30ep_train"),
        "algo": "MA-TransA3C-v3.1b-30ep",
    }
    ma_eval_dir = "MA_v3_1b_30ep_eval"
    ma_algo = "MA-TransA3C-v3.1b-30ep"

    if name == "paper_response_F_p30_arr50_eta4":
        ma_train_kwargs.update({
            "use_lagged_rank_loss": True,
            "lambda_anchor": 0.20,
            "price_anchor": 0.60,
            "use_dual_critic": False,
            "use_pressure_obs": True,
            "pressure_ema_alpha": 0.8,
            "output_dir": str(base / "MA_v8b0_pressureobs_only_train"),
            "algo": "MA-TransA3C-v8b0-pressureobs-only",
        })
        ma_eval_dir = "MA_v8b0_pressureobs_only_eval"
        ma_algo = "MA-TransA3C-v8b0-pressureobs-only"

    _, ma_policy, _ = train_ma_transa3c(
        **ma_train_kwargs,
    )
    evaluate_ma_transa3c(
        ma_policy,
        num_episodes=resolved_eval_episodes,
        steps_per_episode=96,
        seed=2026,
        k_neighbors=4,
        use_ma_station_obs=True,
        use_pressure_obs=ma_train_kwargs.get("use_pressure_obs", False),
        pressure_ema_alpha=ma_train_kwargs.get("pressure_ema_alpha", 0.8),
        env_config=cfg,
        output_dir=str(base / ma_eval_dir),
        algo=ma_algo,
    )

    _write_comparison(
        base,
        q_threshold=float(cfg.get("q_threshold", 4.0)),
        ma_eval_dir=ma_eval_dir,
        ma_label=ma_algo,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only the named response-grid config. Can be repeated.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=30,
        help="Training episodes for each learning baseline.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Evaluation episodes. Defaults to 1 for simulated data and test days for real orders.",
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=5,
        help="Validation episodes used for checkpoint selection.",
    )
    parser.add_argument(
        "--real-order-data-path",
        default=None,
        help="Optional xlsx order file used to replay real arrivals and EV attributes.",
    )
    parser.add_argument(
        "--real-order-sheet-name",
        default="Sheet2",
        help="Worksheet containing order-level rows.",
    )
    parser.add_argument(
        "--real-order-arrival-scale",
        type=float,
        default=1.0,
        help="Scale real arrival counts before injecting them into the simulator.",
    )
    parser.add_argument(
        "--real-order-date-filter",
        default="dominant_month",
        help="Use 'dominant_month', 'all', or a YYYY-MM month from the xlsx.",
    )
    parser.add_argument(
        "--real-order-use-lmp",
        action="store_true",
        help="Use per-kWh electricity fees from real orders as the LMP signal.",
    )
    parser.add_argument(
        "--real-order-train-days",
        type=int,
        default=20,
        help="Number of real-order days assigned to the train split.",
    )
    parser.add_argument(
        "--real-order-validation-days",
        type=int,
        default=5,
        help="Number of real-order days assigned to the validation split.",
    )
    parser.add_argument(
        "--real-order-test-days",
        type=int,
        default=6,
        help="Number of real-order days assigned to the test split.",
    )
    args = parser.parse_args()

    only = set(args.only)
    for name, cfg in iter_paper_response_grid():
        if only and name not in only:
            continue
        print(f"\n===== RUN {name} =====")
        run_one(
            name,
            cfg,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            validation_episodes=args.validation_episodes,
            real_order_data_path=args.real_order_data_path,
            real_order_sheet_name=args.real_order_sheet_name,
            real_order_arrival_scale=args.real_order_arrival_scale,
            real_order_date_filter=args.real_order_date_filter,
            real_order_use_lmp=args.real_order_use_lmp,
            real_order_train_days=args.real_order_train_days,
            real_order_validation_days=args.real_order_validation_days,
            real_order_test_days=args.real_order_test_days,
        )


if __name__ == "__main__":
    main()
