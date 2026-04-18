import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.taxiManager import TaxiManager
from scripts.logs.generate_expert_logs import choose_best_value_target, choose_step_toward


WIDTH = 15
HEIGHT = 15
NUM_PASSENGERS = 2
NUM_EPISODES = 20
MAX_STEPS = 100

MOVE_PENALTY = -0.1
INVALID_MOVE_PENALTY = -1.0
PICKUP_BONUS = 0.0

def run_episode() -> float:
    manager = TaxiManager(width=WIDTH, height=HEIGHT, num_passengers=NUM_PASSENGERS)
    manager.create_passengers()

    total_reward = 0.0
    step_count = 0

    while not manager.is_done() and step_count < MAX_STEPS:
        target_pos, _ = choose_best_value_target(manager)
        dx, dy = choose_step_toward(manager.taxi.position(), target_pos)

        moved = manager.move_taxi(dx, dy)

        picked_up = False
        dropped_off = False
        payout = 0.0

        if moved:
            picked_up = manager.pickup_passenger()
            current_passenger_before_drop = manager.taxi.current_passenger
            dropped_off = manager.dropoff_passenger()
            if dropped_off and current_passenger_before_drop is not None:
                payout = float(current_passenger_before_drop.payout)

        reward = compute_reward(
            moved=moved,
            picked_up=picked_up,
            dropped_off=dropped_off,
            payout=payout,
        )

        total_reward += reward
        step_count += 1

    return total_reward

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

def main() -> None:
    rewards = [run_episode() for _ in range(NUM_EPISODES)]
    average_reward = sum(rewards) / len(rewards)
    print(f"Heuristic average reward over {NUM_EPISODES} episodes: {average_reward:.2f}")


if __name__ == "__main__":
    main()
