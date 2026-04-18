import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import d3rlpy

from environment.taxiManager import TaxiManager


WIDTH = 15
HEIGHT = 15
NUM_EPISODES = 20
NUM_PASSENGERS = 2
MAX_STEPS = 400

MOVE_PENALTY = -0.1
INVALID_MOVE_PENALTY = -1.0
PICKUP_BONUS = 1.0

DATA_DIR = Path("data")
QUALITY_TO_MODEL_FILES = {
    "cql": {
        "expert": DATA_DIR / "offline_rl_cql_expert.d3",
        "mild": DATA_DIR / "offline_rl_cql_mild.d3",
        "mixed": DATA_DIR / "offline_rl_cql_mixed.d3",
        "poor": DATA_DIR / "offline_rl_cql_poor.d3",
    },
    "bcq": {
        "expert": DATA_DIR / "offline_rl_bcq_expert.d3",
        "mild": DATA_DIR / "offline_rl_bcq_mild.d3",
        "mixed": DATA_DIR / "offline_rl_bcq_mixed.d3",
        "poor": DATA_DIR / "offline_rl_bcq_poor.d3",
    },
}
LEGACY_MODEL_FILES = {
    "cql": DATA_DIR / "offline_rl_cql.d3",
    "bcq": DATA_DIR / "offline_rl_bcq.d3",
}

ID_TO_ACTION = {
    0: (0, -1),   # up
    1: (-1, 0),   # left
    2: (0, 1),    # down
    3: (1, 0),    # right
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an offline RL model.")
    parser.add_argument(
        "--algorithm",
        choices=sorted(QUALITY_TO_MODEL_FILES),
        default=None,
        help="Offline RL algorithm to evaluate. If omitted, use CQL if available, otherwise BCQ.",
    )
    parser.add_argument(
        "--quality",
        choices=sorted(next(iter(QUALITY_TO_MODEL_FILES.values()))),
        default=None,
        help="Evaluate the model trained on this data quality. If omitted, use the first matching model found in data.",
    )
    return parser.parse_args()


def find_first_model_for_algorithm(algo_name: str) -> Path | None:
    candidates = sorted(DATA_DIR.glob(f"offline_rl_{algo_name}_*.d3"))

    if candidates:
        return candidates[0]

    legacy_model_file = LEGACY_MODEL_FILES[algo_name]
    if legacy_model_file.exists():
        return legacy_model_file

    return None


def resolve_model_file(
    algo_name: str | None,
    quality: str | None,
) -> tuple[str, Path]:
    if quality is not None and algo_name is None:
        for candidate_algo_name in ("cql", "bcq"):
            model_file = QUALITY_TO_MODEL_FILES[candidate_algo_name][quality]
            if model_file.exists():
                return candidate_algo_name, model_file

        raise FileNotFoundError(
            f"No offline RL model found for quality='{quality}'. "
            "Checked CQL first, then BCQ."
        )

    if algo_name is not None and quality is not None:
        model_file = QUALITY_TO_MODEL_FILES[algo_name][quality]
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model for algorithm='{algo_name}' and quality='{quality}' "
                f"not found: {model_file}"
            )
        return algo_name, model_file

    if algo_name is not None:
        model_file = find_first_model_for_algorithm(algo_name)
        if model_file is None:
            raise FileNotFoundError(
                f"No model found for algorithm='{algo_name}'. "
                f"Expected data/offline_rl_{algo_name}_*.d3"
            )
        return algo_name, model_file

    for candidate_algo_name in ("cql", "bcq"):
        model_file = find_first_model_for_algorithm(candidate_algo_name)
        if model_file is not None:
            return candidate_algo_name, model_file

    raise FileNotFoundError(
        "No offline RL model found. Expected data/offline_rl_cql_*.d3 "
        "or data/offline_rl_bcq_*.d3."
    )


def build_state(manager: TaxiManager) -> list[float]:
    state: list[float] = [
        float(manager.taxi.x),
        float(manager.taxi.y),
        1.0 if manager.taxi.current_passenger is not None else 0.0,
    ]

    for passenger in manager.passengers:
        state.extend(
            [
                1.0 if passenger.picked_up else 0.0,
                1.0 if passenger.completed else 0.0,
                float(passenger.pickup_x),
                float(passenger.pickup_y),
                float(passenger.dropoff_x),
                float(passenger.dropoff_y),
                float(passenger.payout),
            ]
        )

    return state

def compute_reward(
    moved: bool,
    picked_up: bool,
    dropped_off: bool,
    payout: float,
) -> float:
    if not moved:
        return INVALID_MOVE_PENALTY

    reward = MOVE_PENALTY

    if picked_up:
        reward += PICKUP_BONUS

    if dropped_off:
        reward += payout

    return reward

def run_episode(algo) -> dict[str, float | int | bool]:
    manager = TaxiManager(width=WIDTH, height=HEIGHT, num_passengers=NUM_PASSENGERS)
    manager.create_passengers()

    total_reward = 0.0
    max_steps = MAX_STEPS
    step_count = 0

    invalid_moves = 0
    pickup_count = 0
    dropoff_count = 0

    while not manager.is_done() and step_count < max_steps:
        state = np.asarray([build_state(manager)], dtype=np.float32)
        action_id = int(algo.predict(state)[0])

        dx, dy = ID_TO_ACTION[action_id]

        moved = manager.move_taxi(dx, dy)

        picked_up = False
        dropped_off = False
        payout = 0.0

        if not moved:
            invalid_moves += 1

        if moved:
            picked_up = manager.pickup_passenger()
            if picked_up:
                pickup_count += 1

            current_passenger_before_drop = manager.taxi.current_passenger
            dropped_off = manager.dropoff_passenger()
            if dropped_off:
                dropoff_count += 1
                if current_passenger_before_drop is not None:
                    payout = float(current_passenger_before_drop.payout)

        reward = compute_reward(
            moved=moved,
            picked_up=picked_up,
            dropped_off=dropped_off,
            payout=payout,
        )

        total_reward += reward
        step_count += 1

    success = manager.is_done()

    return {
        "reward": total_reward,
        "success": success,
        "episode_length": step_count,
        "pickup_count": pickup_count,
        "dropoff_count": dropoff_count,
        "invalid_moves": invalid_moves,
        "invalid_move_rate": (invalid_moves / step_count) if step_count > 0 else 0.0,
        "picked_up_any": pickup_count > 0,
        "dropped_off_any": dropoff_count > 0,
    }


def main() -> None:
    args = parse_args()
    algo_name, model_file = resolve_model_file(args.algorithm, args.quality)

    device = "cuda:0" if torch.cuda.is_available() else False
    algo = d3rlpy.load_learnable(str(model_file), device=device)

    results = [run_episode(algo) for _ in range(NUM_EPISODES)]

    average_reward = sum(r["reward"] for r in results) / NUM_EPISODES
    success_rate = sum(1 for r in results if r["success"]) / NUM_EPISODES
    average_episode_length = sum(r["episode_length"] for r in results) / NUM_EPISODES
    pickup_rate = sum(1 for r in results if r["picked_up_any"]) / NUM_EPISODES
    dropoff_rate = sum(1 for r in results if r["dropped_off_any"]) / NUM_EPISODES
    average_invalid_move_rate = sum(r["invalid_move_rate"] for r in results) / NUM_EPISODES

    average_pickups_per_episode = sum(r["pickup_count"] for r in results) / NUM_EPISODES
    average_dropoffs_per_episode = sum(r["dropoff_count"] for r in results) / NUM_EPISODES
    average_invalid_moves_per_episode = sum(r["invalid_moves"] for r in results) / NUM_EPISODES

    print(f"Algorithm: {algo_name.upper()}")
    print(f"Model file: {model_file}")
    print(f"Rollout evaluation over {NUM_EPISODES} episodes")
    print(f"Average reward: {average_reward:.2f}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average episode length: {average_episode_length:.2f}")
    print(f"Pickup rate: {pickup_rate:.2%}")
    print(f"Dropoff rate: {dropoff_rate:.2%}")
    print(f"Average invalid move rate: {average_invalid_move_rate:.2%}")
    print(f"Average pickups per episode: {average_pickups_per_episode:.2f}")
    print(f"Average dropoffs per episode: {average_dropoffs_per_episode:.2f}")
    print(f"Average invalid moves per episode: {average_invalid_moves_per_episode:.2f}")


if __name__ == "__main__":
    main()
