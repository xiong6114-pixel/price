from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from case_studies.power.ev_public_charging_case.train_rllib import (
    create_charging_env,
    get_paper_response_F_p30_arr50_eta4_config,
    run_fixed_pricing,
)


def _hash_rows(rows) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_fp(seed: int, output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _, step_logs = run_fixed_pricing(
        num_episodes=1,
        steps_per_episode=24,
        seed=seed,
        env_config=get_paper_response_F_p30_arr50_eta4_config(),
        output_dir=str(output_dir),
    )
    return step_logs


def main() -> int:
    failures = []
    base_dir = Path("D:/price/outputs/debug_repro_checks")
    base_dir.mkdir(parents=True, exist_ok=True)

    step_logs_a = _run_fp(seed=2026, output_dir=base_dir / "same_seed_a")
    step_logs_b = _run_fp(seed=2026, output_dir=base_dir / "same_seed_b")
    hash_a = _hash_rows(step_logs_a)
    hash_b = _hash_rows(step_logs_b)
    if hash_a != hash_b:
        failures.append(f"same-seed mismatch: {hash_a} != {hash_b}")

    step_logs_c = _run_fp(seed=2027, output_dir=base_dir / "different_seed")
    hash_c = _hash_rows(step_logs_c)
    if hash_a == hash_c:
        failures.append("different-seed run unexpectedly matched same-seed hash")

    env = create_charging_env(get_paper_response_F_p30_arr50_eta4_config())
    env.reset(seed=123)
    env_state = env.global_state_to_env_state(env.global_state)
    station_id = next(iter(env.station_positions))
    env._station_queues[station_id] = []
    for slot_id, mapped_station_id in env_state.slot_to_station.items():
        if mapped_station_id == station_id:
            slot_state = env_state.slot_states[slot_id]
            slot_state.open_or_not = 1
            slot_state.occupied = 1
    wait_min = env._estimate_wait_min(env_state, station_id)
    if wait_min <= 0.0:
        failures.append(f"full-station wait should be > 0, got {wait_min}")

    total_profit = 0.0
    total_revenue = 0.0
    total_grid_cost = 0.0
    for row in step_logs_a:
        profit = float(row["profit"])
        revenue = float(row["revenue"])
        grid_cost = float(row["grid_cost"])
        congestion_penalty = float(row["congestion_penalty"])
        reward = float(row["reward"])
        if abs((profit - congestion_penalty) - reward) > 1e-6:
            failures.append(
                "reward/accounting mismatch at "
                f"episode={row['episode']} step={row['step']} station={row['station']}"
            )
            break
        total_profit += profit
        total_revenue += revenue
        total_grid_cost += grid_cost
    if abs(total_profit - (total_revenue - total_grid_cost)) > 1e-6:
        failures.append(
            "total accounting mismatch: "
            f"profit={total_profit} revenue-grid={total_revenue - total_grid_cost}"
        )

    print(f"same_seed_hash={hash_a}")
    print(f"different_seed_hash={hash_c}")
    print(f"full_station_wait_min={wait_min}")
    print(f"total_profit={total_profit:.6f}")
    print(f"total_revenue_minus_grid={total_revenue - total_grid_cost:.6f}")

    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
