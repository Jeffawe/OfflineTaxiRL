import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.taxiManager import TaxiManager
from scripts.generate_logs import choose_best_value_target, choose_step_toward


WIDTH = 10
HEIGHT = 10
NUM_PASSENGERS = 2
NUM_EPISODES = 20
MAX_STEPS = 100

MOVE_PENALTY = -0.1
INVALID_MOVE_PENALTY = -1.0


def run_episode() -> float:
    manager = TaxiManager(width=WIDTH, height=HEIGHT, num_passengers=NUM_PASSENGERS)
    manager.create_passengers()

    total_reward = 0.0
    step_count = 0

    while not manager.is_done() and step_count < MAX_STEPS:
        target_pos, _ = choose_best_value_target(manager)
        dx, dy = choose_step_toward(manager.taxi.position(), target_pos)

        moved = manager.move_taxi(dx, dy)
        dropped_off = False
        payout = 0.0

        if moved:
            manager.pickup_passenger()
            current_passenger_before_drop = manager.taxi.current_passenger
            dropped_off = manager.dropoff_passenger()
            if dropped_off and current_passenger_before_drop is not None:
                payout = float(current_passenger_before_drop.payout)

        reward = INVALID_MOVE_PENALTY if not moved else MOVE_PENALTY
        if dropped_off:
            reward += payout

        total_reward += reward
        step_count += 1

    return total_reward


def main() -> None:
    rewards = [run_episode() for _ in range(NUM_EPISODES)]
    average_reward = sum(rewards) / len(rewards)
    print(f"Heuristic average reward over {NUM_EPISODES} episodes: {average_reward:.2f}")


if __name__ == "__main__":
    main()
