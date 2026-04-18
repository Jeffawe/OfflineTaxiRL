import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import d3rlpy

from environment.taxiManager import TaxiManager


WIDTH = 30
HEIGHT = 30
NUM_EPISODES = 20
NUM_PASSENGERS = 2
MAX_STEPS = 400

MOVE_PENALTY = -0.1
INVALID_MOVE_PENALTY = -1.0
PICKUP_BONUS = 1.0

MODEL_FILES = {
    "cql": Path("data/offline_rl_cql.d3"),
    "bcq": Path("data/offline_rl_bcq.d3")
}

ID_TO_ACTION = {
    0: (0, -1),   # up
    1: (-1, 0),   # left
    2: (0, 1),    # down
    3: (1, 0),    # right
}


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

def parse_algorithm_name() -> str:
    if len(sys.argv) < 2:
        return "cql"

    algo_name = sys.argv[1].strip().lower()

    if algo_name not in MODEL_FILES:
        valid_names = ", ".join(MODEL_FILES.keys())
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Expected one of: {valid_names}"
        )

    return algo_name

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
    algo_name = parse_algorithm_name()
    model_file = MODEL_FILES[algo_name]

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file for '{algo_name}' not found: {model_file}"
        )

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