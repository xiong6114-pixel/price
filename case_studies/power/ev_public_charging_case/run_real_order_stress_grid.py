"""Run real-order-calibrated I/MA stress experiments.

This runner treats the order workbook/daily CSVs as a company-level network
arrival stream. Station arrivals, routing, queues, and violations remain virtual
simulation outputs because the source orders do not contain station IDs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from case_studies.power.ev_public_charging_case.compare_baselines import (
    load_one_baseline,
    write_csv,
)
from case_studies.power.ev_public_charging_case.train_rllib import (
    evaluate_independent_transa3c,
    evaluate_ma_transa3c,
    get_paper_response_F_p30_arr50_eta4_config,
    train_independent_transa3c,
    train_ma_transa3c,
)


METHOD_I = "I"
METHOD_MA = "MA"
ALGO_I = "I-TransA3C"
ALGO_MA = "MA-TransA3C-v8b0-pressureobs-only"


def _parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(part) for part in _parse_csv_list(value)]


def _load_episode_rows(path: Path) -> List[Dict[str, str]]:
    episode_path = path / "episode_metrics.csv"
    if not episode_path.exists():
        return []
    with episode_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_nan(value: Any) -> float:
    if value in {None, ""}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _last_nonempty(rows: List[Dict[str, str]], key: str) -> Any:
    for row in reversed(rows):
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return ""


def _training_summary(train_dir: Path) -> Dict[str, Any]:
    rows = _load_episode_rows(train_dir)
    if not rows:
        return {
            "best_validation_episode": math.nan,
            "best_validation_score": math.nan,
            "actor_mean_price_last5": math.nan,
            "train_policy_device": "",
        }

    best_episode = _last_nonempty(rows, "best_validation_episode")
    best_score = _last_nonempty(rows, "best_validation_score")
    if best_episode == "" or best_score == "":
        validation_rows = [
            row for row in rows
            if row.get("validation_score") not in {None, ""}
        ]
        if validation_rows:
            best_row = max(
                validation_rows,
                key=lambda row: _float_or_nan(row.get("validation_score")),
            )
            best_episode = best_row.get("episode", "")
            best_score = best_row.get("validation_score", "")

    actor_prices = [
        _float_or_nan(row.get("actor_mean_price"))
        for row in rows
        if math.isfinite(_float_or_nan(row.get("actor_mean_price")))
    ]
    actor_last5 = mean(actor_prices[-5:]) if actor_prices else math.nan

    return {
        "best_validation_episode": _float_or_nan(best_episode),
        "best_validation_score": _float_or_nan(best_score),
        "actor_mean_price_last5": actor_last5,
        "train_policy_device": _last_nonempty(rows, "device"),
    }


def _check_device(device: str, require_gpu: bool) -> None:
    if require_gpu and device != "cuda":
        raise RuntimeError("--require-gpu requires --device cuda.")
    if device != "cuda":
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build before running this experiment."
        )
    print(
        f"[GPU] Using {torch.cuda.get_device_name(0)} | "
        f"torch={torch.__version__} | cuda={torch.version.cuda}"
    )


def _env_config(args: argparse.Namespace, target_orders_per_day: int) -> Dict[str, Any]:
    cfg = get_paper_response_F_p30_arr50_eta4_config()
    cfg.update({
        "real_order_daily_dir": str(Path(args.real_order_daily_dir)),
        "real_order_sampling_mode": args.real_order_sampling_mode,
        "real_order_target_orders_per_day": int(target_orders_per_day),
        "real_order_date_filter": args.real_order_date_filter,
        "real_order_min_energy_kwh": args.real_order_min_energy_kwh,
        "real_order_train_days": args.real_order_train_days,
        "real_order_validation_days": args.real_order_validation_days,
        "real_order_test_days": args.real_order_test_days,
    })
    return cfg


def _method_dir_name(method: str) -> str:
    return "I_TransA3C" if method == METHOD_I else "MA_v8b0_pressureobs_only"


def _train_method(
    method: str,
    cfg: Dict[str, Any],
    train_dir: Path,
    args: argparse.Namespace,
):
    if method == METHOD_I:
        _env, policies, _returns = train_independent_transa3c(
            num_episodes=args.train_episodes,
            steps_per_episode=args.steps_per_episode,
            validation_every=args.validation_every,
            validation_num_episodes=args.validation_episodes,
            env_config=cfg,
            device=args.device,
            output_dir=str(train_dir),
            algo=ALGO_I,
        )
        return policies

    _env, policy, _returns = train_ma_transa3c(
        num_episodes=args.train_episodes,
        steps_per_episode=args.steps_per_episode,
        validation_every=args.validation_every,
        validation_num_episodes=args.validation_episodes,
        k_neighbors=4,
        use_ma_station_obs=True,
        use_lagged_rank_loss=True,
        lambda_anchor=0.20,
        price_anchor=0.60,
        use_dual_critic=False,
        use_pressure_obs=True,
        pressure_ema_alpha=0.8,
        env_config=cfg,
        device=args.device,
        output_dir=str(train_dir),
        algo=ALGO_MA,
    )
    return policy


def _evaluate_method(
    method: str,
    policy_or_policies,
    cfg: Dict[str, Any],
    eval_dir: Path,
    args: argparse.Namespace,
) -> None:
    if method == METHOD_I:
        evaluate_independent_transa3c(
            policy_or_policies,
            num_episodes=args.eval_episodes,
            steps_per_episode=args.steps_per_episode,
            env_config=cfg,
            output_dir=str(eval_dir),
            algo=ALGO_I,
        )
        return

    evaluate_ma_transa3c(
        policy_or_policies,
        num_episodes=args.eval_episodes,
        steps_per_episode=args.steps_per_episode,
        k_neighbors=4,
        use_ma_station_obs=True,
        use_pressure_obs=True,
        pressure_ema_alpha=0.8,
        env_config=cfg,
        output_dir=str(eval_dir),
        algo=ALGO_MA,
    )


def _summarize_eval(
    method: str,
    protocol: str,
    train_target: int,
    eval_target: int,
    train_dir: Path,
    eval_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row = load_one_baseline(
        f"{method}-{protocol}-train{train_target}-eval{eval_target}",
        eval_dir,
        q_threshold=float(get_paper_response_F_p30_arr50_eta4_config().get("q_threshold", 4.0)),
    )
    row.update({
        "method": method,
        "protocol": protocol,
        "train_target_orders_per_day": train_target,
        "eval_target_orders_per_day": eval_target,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "real_order_sampling_mode": args.real_order_sampling_mode,
        "train_output_dir": str(train_dir),
        "eval_output_dir": str(eval_dir),
    })
    row.update(_training_summary(train_dir))
    return row


def _run_training_block(
    methods: Iterable[str],
    protocol: str,
    train_target: int,
    eval_targets: Iterable[int],
    base_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    train_cfg = _env_config(args, train_target)

    for method in methods:
        method_dir = _method_dir_name(method)
        train_dir = base_dir / method_dir / "train"
        print(f"[RUN] {protocol} {method} train_target={train_target} -> {train_dir}")
        policy_or_policies = _train_method(method, train_cfg, train_dir, args)

        for eval_target in eval_targets:
            eval_cfg = _env_config(args, int(eval_target))
            eval_dir = base_dir / method_dir / f"eval_target{int(eval_target)}"
            print(f"[EVAL] {protocol} {method} eval_target={eval_target} -> {eval_dir}")
            _evaluate_method(method, policy_or_policies, eval_cfg, eval_dir, args)
            rows.append(
                _summarize_eval(
                    method=method,
                    protocol=protocol,
                    train_target=train_target,
                    eval_target=int(eval_target),
                    train_dir=train_dir,
                    eval_dir=eval_dir,
                    args=args,
                )
            )

    return rows


def run(args: argparse.Namespace) -> None:
    methods = [method.upper() for method in _parse_csv_list(args.methods)]
    invalid = sorted(set(methods) - {METHOD_I, METHOD_MA})
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Use I and/or MA.")
    _check_device(args.device, args.require_gpu)

    base_target = int(args.target_orders_per_day)
    stress_targets = _parse_int_list(args.stress_target_orders)
    all_targets = [base_target] + [target for target in stress_targets if target != base_target]
    output_root = Path(args.output_root)
    device_tag = "gpu" if args.device == "cuda" else args.device.replace(":", "_")
    summary_rows: List[Dict[str, Any]] = []

    if args.protocol in {"robustness", "both"}:
        base_dir = output_root / f"realorders_slot{base_target}_{args.train_episodes}ep_{device_tag}"
        summary_rows.extend(
            _run_training_block(
                methods=methods,
                protocol="robustness",
                train_target=base_target,
                eval_targets=all_targets,
                base_dir=base_dir,
                args=args,
            )
        )

    if args.protocol in {"retrain", "both"}:
        for target in all_targets:
            base_dir = output_root / f"realorders_retrain_slot{target}_{args.train_episodes}ep_{device_tag}"
            summary_rows.extend(
                _run_training_block(
                    methods=methods,
                    protocol="retrain",
                    train_target=target,
                    eval_targets=[target],
                    base_dir=base_dir,
                    args=args,
                )
            )

    out_path = output_root / "real_order_stress_comparison_summary.csv"
    write_csv(out_path, summary_rows)
    print(f"[DONE] Summary saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="I,MA")
    parser.add_argument(
        "--real-order-daily-dir",
        default="D:/price/outputs/real_order_daily_split_strict/by_start_date_clean",
    )
    parser.add_argument("--real-order-sampling-mode", default="slot_profile_matched")
    parser.add_argument("--target-orders-per-day", type=int, default=1200)
    parser.add_argument("--stress-target-orders", default="1600,2400")
    parser.add_argument("--real-order-date-filter", default="2024-01")
    parser.add_argument("--real-order-min-energy-kwh", type=float, default=0.5)
    parser.add_argument("--real-order-train-days", type=int, default=20)
    parser.add_argument("--real-order-validation-days", type=int, default=5)
    parser.add_argument("--real-order-test-days", type=int, default=6)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=96)
    parser.add_argument(
        "--protocol",
        choices=("robustness", "retrain", "both"),
        default="both",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output-root", default="D:/price/outputs")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
