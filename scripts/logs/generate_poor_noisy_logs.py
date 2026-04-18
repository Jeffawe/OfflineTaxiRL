import json
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from generate_expert_logs import (
    ACTION_TO_ID,
    ACTION_TO_NAME,
    BASE_SEED,
    FLUSH_TRANSITIONS_EVERY,
    MAX_STEPS_PER_EPISODE,
    NUM_EPISODES,
    PROGRESS_EVERY,
    build_state,
    choose_best_value_target,
    choose_step_toward,
    compute_reward,
    create_manager,
    find_passenger_id,
    flush_transition_chunk,
    snapshot_passengers,
)


RAW_LOG_FILE = Path("data/raw_taxi_logs_poor_noisy.jsonl")
TRANSITIONS_FILE = Path("data/transitions_poor_noisy.pkl")
POLICY_NAME = "poor_noisy_v1"
ACTIONS = list(ACTION_TO_ID.keys())

POLICY_MIX = {
    "best_value": 0.20,
    "noisy_best_value": 0.30,
    "random": 0.50,
}
NOISY_BEST_VALUE_RANDOM_ACTION_PROBABILITY = 0.50


def ensure_data_dir() -> None:
    RAW_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRANSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


def sample_episode_policy() -> str:
    threshold = random.random()
    cumulative = 0.0

    for policy_name, probability in POLICY_MIX.items():
        cumulative += probability
        if threshold <= cumulative:
            return policy_name

    return "random"


def choose_action(
    episode_policy: str,
    taxi_pos_before: tuple[int, int],
    target_pos: tuple[int, int],
) -> tuple[tuple[int, int], str]:
    if episode_policy == "random":
        return random.choice(ACTIONS), "random_policy"

    if (
        episode_policy == "noisy_best_value"
        and random.random() < NOISY_BEST_VALUE_RANDOM_ACTION_PROBABILITY
    ):
        return random.choice(ACTIONS), "random_noise"

    return choose_step_toward(taxi_pos_before, target_pos), "best_value"


def generate_logs() -> None:
    ensure_data_dir()

    sim_time = datetime(2026, 1, 1, 8, 0, 0)

    total_events = 0
    total_transitions = 0
    transitions_chunk: list[dict[str, Any]] = []

    with RAW_LOG_FILE.open("w", encoding="utf-8") as raw_f, TRANSITIONS_FILE.open("wb") as trans_f:
        for episode_id in range(NUM_EPISODES):
            manager = create_manager(BASE_SEED + episode_id)
            episode_policy = sample_episode_policy()

            step_idx = 0
            done = manager.is_done()

            while not done and step_idx < MAX_STEPS_PER_EPISODE:
                state = build_state(manager)

                taxi_pos_before = manager.taxi.position()
                carrying_before = find_passenger_id(manager, manager.taxi.current_passenger)

                target_pos, heuristic_meta = choose_best_value_target(manager)
                (dx, dy), action_source = choose_action(episode_policy, taxi_pos_before, target_pos)

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

                next_state = build_state(manager)
                done = manager.is_done() or (step_idx + 1 >= MAX_STEPS_PER_EPISODE)

                taxi_pos_after = manager.taxi.position()
                carrying_after = find_passenger_id(manager, manager.taxi.current_passenger)

                action_id = ACTION_TO_ID[(dx, dy)]
                action_name = ACTION_TO_NAME[(dx, dy)]

                raw_event = {
                    "event_type": "dispatch_step",
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "timestamp": sim_time.isoformat(),
                    "taxi_id": "taxi_0",
                    "policy_name": POLICY_NAME,
                    "episode_policy": episode_policy,
                    "action_source": action_source,
                    "policy_mix": POLICY_MIX,
                    "noisy_best_value_random_action_probability": NOISY_BEST_VALUE_RANDOM_ACTION_PROBABILITY,

                    "taxi_x_before": taxi_pos_before[0],
                    "taxi_y_before": taxi_pos_before[1],
                    "taxi_x_after": taxi_pos_after[0],
                    "taxi_y_after": taxi_pos_after[1],

                    "carrying_before": carrying_before,
                    "carrying_after": carrying_after,

                    "action_id": action_id,
                    "action_name": action_name,
                    "dx": dx,
                    "dy": dy,

                    "moved": moved,
                    "picked_up": picked_up,
                    "dropped_off": dropped_off,
                    "payout": payout,
                    "reward": reward,
                    "done": done,

                    "heuristic_decision_mode": heuristic_meta["decision_mode"],
                    "target_passenger_id": heuristic_meta["target_passenger_id"],
                    "target_x": heuristic_meta["target_x"],
                    "target_y": heuristic_meta["target_y"],
                    "pickup_distance_est": heuristic_meta["pickup_distance"],
                    "trip_distance_est": heuristic_meta["trip_distance"],
                    "score": heuristic_meta["score"],
                    "expected_payout": heuristic_meta["payout"],

                    "state": state,
                    "next_state": next_state,
                    "passengers_snapshot": snapshot_passengers(manager),
                }
                raw_f.write(json.dumps(raw_event) + "\n")
                total_events += 1

                transitions_chunk.append(
                    {
                        "episode_id": episode_id,
                        "step_idx": step_idx,
                        "state": state,
                        "action": action_id,
                        "action_name": action_name,
                        "action_dxdy": (dx, dy),
                        "reward": reward,
                        "next_state": next_state,
                        "done": done,
                    }
                )
                total_transitions += 1

                step_idx += 1
                sim_time += timedelta(seconds=30)

            if (episode_id + 1) % FLUSH_TRANSITIONS_EVERY == 0:
                flush_transition_chunk(transitions_chunk, trans_f)
                trans_f.flush()

            if (
                (episode_id + 1) % PROGRESS_EVERY == 0
                or (episode_id + 1) == NUM_EPISODES
            ):
                print(
                    f"Generated {episode_id + 1}/{NUM_EPISODES} episodes "
                    f"({((episode_id + 1) / NUM_EPISODES) * 100:.1f}%) | "
                    f"events so far: {total_events} | "
                    f"transitions so far: {total_transitions}"
                )

        flush_transition_chunk(transitions_chunk, trans_f)

    print(f"Saved raw logs to: {RAW_LOG_FILE}")
    print(f"Saved transition chunks to: {TRANSITIONS_FILE}")
    print(f"Total events: {total_events}")
    print(f"Total transitions: {total_transitions}")


if __name__ == "__main__":
    generate_logs()
