import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import d3rlpy

from environment.taxiManager import TaxiManager


MODEL_FILE = Path("data/offline_rl_cql.d3")


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


def run_episode(algo) -> float:
    manager = TaxiManager(width=20, height=20, num_passengers=2)
    manager.create_passengers()

    total_reward = 0.0
    max_steps = 100
    step_count = 0

    while not manager.is_done() and step_count < max_steps:
        state = np.asarray([build_state(manager)], dtype=np.float32)
        action_id = int(algo.predict(state)[0])

        if action_id == 0:
            dx, dy = (0, -1)
        elif action_id == 1:
            dx, dy = (-1, 0)
        elif action_id == 2:
            dx, dy = (0, 1)
        else:
            dx, dy = (1, 0)

        moved = manager.move_taxi(dx, dy)
        dropped_off = False
        payout = 0.0

        if moved:
            manager.pickup_passenger()
            current_passenger_before_drop = manager.taxi.current_passenger
            dropped_off = manager.dropoff_passenger()
            if dropped_off and current_passenger_before_drop is not None:
                payout = float(current_passenger_before_drop.payout)

        reward = -1.0 if not moved else -0.1
        if dropped_off:
            reward += payout

        total_reward += reward
        step_count += 1

    return total_reward


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else False
    algo = d3rlpy.load_learnable(str(MODEL_FILE), device=device)

    rewards = [run_episode(algo) for _ in range(20)]
    average_reward = sum(rewards) / len(rewards)
    print(f"Offline RL average reward over 20 episodes: {average_reward:.2f}")


if __name__ == "__main__":
    main()
