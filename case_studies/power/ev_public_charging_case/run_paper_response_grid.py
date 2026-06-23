from pathlib import Path
import sys

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


def _write_comparison(base: Path, q_threshold: float) -> None:
    rows = [
        load_one_baseline("FP", base / "FP_eval", q_threshold=q_threshold),
        load_one_baseline("I-TransA3C", base / "I_TransA3C_eval", q_threshold=q_threshold),
        load_one_baseline("MAPPO", base / "MAPPO_MLP_eval", q_threshold=q_threshold),
        load_one_baseline("MA-TransA3C", base / "MA_v3_1b_30ep_eval", q_threshold=q_threshold),
    ]
    write_csv(base / "paper_table2_reproduction.csv", add_relative_metrics(rows, fp_algo_name="FP"))


def run_one(name, cfg):
    base = Path("D:/price/outputs") / f"paper_response_{name}"
    base.mkdir(parents=True, exist_ok=True)

    run_fixed_pricing(
        num_episodes=1,
        steps_per_episode=96,
        seed=2026,
        env_config=cfg,
        output_dir=str(base / "FP_eval"),
    )

    _, i_policies, _ = train_independent_transa3c(
        num_episodes=30,
        steps_per_episode=96,
        seed=42,
        gamma=0.99,
        seq_len=8,
        env_config=cfg,
        output_dir=str(base / "I_TransA3C_train"),
        algo="I-TransA3C",
    )
    evaluate_independent_transa3c(
        i_policies,
        num_episodes=1,
        steps_per_episode=96,
        seed=2026,
        seq_len=8,
        env_config=cfg,
        output_dir=str(base / "I_TransA3C_eval"),
        algo="I-TransA3C",
    )

    _, mappo_policy, _ = train_mappo_mlp(
        num_episodes=30,
        steps_per_episode=96,
        seed=42,
        gamma=0.99,
        use_ma_station_obs=True,
        env_config=cfg,
        output_dir=str(base / "MAPPO_MLP_train"),
        algo="MAPPO-MLP",
    )
    evaluate_mappo_mlp(
        mappo_policy,
        num_episodes=1,
        steps_per_episode=96,
        seed=2026,
        use_ma_station_obs=True,
        env_config=cfg,
        output_dir=str(base / "MAPPO_MLP_eval"),
        algo="MAPPO-MLP",
    )

    _, ma_policy, _ = train_ma_transa3c(
        num_episodes=30,
        steps_per_episode=96,
        seed=42,
        gamma=0.99,
        k_neighbors=4,
        use_ma_station_obs=True,
        use_lagged_rank_loss=False,
        lambda_rank=0.20,
        rank_margin=0.02,
        rank_eps=0.10,
        env_config=cfg,
        output_dir=str(base / "MA_v3_1b_30ep_train"),
        algo="MA-TransA3C-v3.1b-30ep",
    )
    evaluate_ma_transa3c(
        ma_policy,
        num_episodes=1,
        steps_per_episode=96,
        seed=2026,
        k_neighbors=4,
        use_ma_station_obs=True,
        env_config=cfg,
        output_dir=str(base / "MA_v3_1b_30ep_eval"),
        algo="MA-TransA3C-v3.1b-30ep",
    )

    _write_comparison(base, q_threshold=float(cfg.get("q_threshold", 4.0)))


def main():
    for name, cfg in iter_paper_response_grid():
        print(f"\n===== RUN {name} =====")
        run_one(name, cfg)


if __name__ == "__main__":
    main()
