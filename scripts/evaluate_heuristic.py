import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import statistics

from environment.taxiManager import TaxiManager
from scripts.logs.generate_expert_logs import choose_best_value_target, choose_step_toward


WIDTH = 15
HEIGHT = 15
NUM_PASSENGERS = 2
NUM_EPISODES = 100
EVAL_SEEDS = [0, 1, 2, 3, 4]
MAX_STEPS = 400

MOVE_PENALTY = -0.1
INVALID_MOVE_PENALTY = -1.0
PICKUP_BONUS = 1.0

def run_episode() -> dict[str, float | int | bool]:
    manager = TaxiManager(width=WIDTH, height=HEIGHT, num_passengers=NUM_PASSENGERS)
    manager.create_passengers()

    total_reward = 0.0
    step_count = 0
    invalid_moves = 0
    pickup_count = 0
    dropoff_count = 0

    while not manager.is_done() and step_count < MAX_STEPS:
        target_pos, _ = choose_best_value_target(manager)
        dx, dy = choose_step_toward(manager.taxi.position(), target_pos)

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

def evaluate_seed(seed: int) -> dict[str, float]:
    random.seed(seed)
    results = [run_episode() for _ in range(NUM_EPISODES)]

    return compute_summary(results)


def compute_summary(results: list[dict[str, float | int | bool]]) -> dict[str, float]:
    num_episodes = len(results)

    return {
        "average_reward": sum(float(r["reward"]) for r in results) / num_episodes,
        "success_rate": sum(1 for r in results if r["success"]) / num_episodes,
        "average_episode_length": sum(float(r["episode_length"]) for r in results) / num_episodes,
        "pickup_rate": sum(1 for r in results if r["picked_up_any"]) / num_episodes,
        "dropoff_rate": sum(1 for r in results if r["dropped_off_any"]) / num_episodes,
        "average_invalid_move_rate": sum(float(r["invalid_move_rate"]) for r in results) / num_episodes,
        "average_pickups_per_episode": sum(float(r["pickup_count"]) for r in results) / num_episodes,
        "average_dropoffs_per_episode": sum(float(r["dropoff_count"]) for r in results) / num_episodes,
        "average_invalid_moves_per_episode": sum(float(r["invalid_moves"]) for r in results) / num_episodes,
    }


def format_values(
    summaries: list[dict[str, float]],
    metric: str,
    percent: bool = False,
) -> str:
    values = [summary[metric] for summary in summaries]

    if percent:
        return ", ".join(f"{value:.2%}" for value in values)

    return ", ".join(f"{value:.2f}" for value in values)


def main() -> None:
    summaries = [evaluate_seed(seed) for seed in EVAL_SEEDS]
    success_rates = [summary["success_rate"] for summary in summaries]

    print("Algorithm: HEURISTIC")
    print(f"Rollout evaluation over {NUM_EPISODES} episodes for each seed")
    print(f"Seeds: {', '.join(str(seed) for seed in EVAL_SEEDS)}")
    print(f"Average reward: {format_values(summaries, 'average_reward')}")
    print(f"Success rate: {format_values(summaries, 'success_rate', percent=True)}")
    print(f"Average episode length: {format_values(summaries, 'average_episode_length')}")
    print(f"Pickup rate: {format_values(summaries, 'pickup_rate', percent=True)}")
    print(f"Dropoff rate: {format_values(summaries, 'dropoff_rate', percent=True)}")
    print(f"Average invalid move rate: {format_values(summaries, 'average_invalid_move_rate', percent=True)}")
    print(f"Average pickups per episode: {format_values(summaries, 'average_pickups_per_episode')}")
    print(f"Average dropoffs per episode: {format_values(summaries, 'average_dropoffs_per_episode')}")
    print(f"Average invalid moves per episode: {format_values(summaries, 'average_invalid_moves_per_episode')}")
    print(f"Mean success rate: {statistics.mean(success_rates):.2%}")
    print(f"Success rate standard deviation: {statistics.stdev(success_rates):.2%}")


if __name__ == "__main__":
    main()
