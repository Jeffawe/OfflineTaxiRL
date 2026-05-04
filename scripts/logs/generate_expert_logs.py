import json
import pickle
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environment.taxiManager import TaxiManager


# ============================================================
# CONFIG
# ============================================================

WIDTH = 15
HEIGHT = 15
NUM_PASSENGERS = 2
NUM_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 400

PROGRESS_EVERY = 1000
FLUSH_TRANSITIONS_EVERY = 1000  # flush transition chunks every N episodes

RAW_LOG_FILE = Path("data/raw_taxi_logs_best_value.jsonl")
TRANSITIONS_FILE = Path("data/transitions_best_value.pkl")

BASE_SEED = 42

# Reward assumptions for training data
MOVE_PENALTY = -0.1
INVALID_MOVE_PENALTY = -1.0
PICKUP_BONUS = 1.0

# Best-value heuristic weights
ALPHA_PICKUP = 0.8
BETA_TRIP = 0.35

# Action mapping for later BC / RL
ACTION_TO_ID = {
    (0, -1): 0,   # up
    (-1, 0): 1,   # left
    (0, 1): 2,    # down
    (1, 0): 3,    # right
}

ACTION_TO_NAME = {
    (0, -1): "UP",
    (-1, 0): "LEFT",
    (0, 1): "DOWN",
    (1, 0): "RIGHT",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_data_dir() -> None:
    RAW_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRANSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_passenger_id(index: int) -> str:
    return f"passenger_{index}"


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


def snapshot_passengers(manager: TaxiManager) -> list[dict[str, Any]]:
    snapshot = []

    for i, passenger in enumerate(manager.passengers):
        snapshot.append(
            {
                "passenger_id": get_passenger_id(i),
                "pickup_x": passenger.pickup_x,
                "pickup_y": passenger.pickup_y,
                "dropoff_x": passenger.dropoff_x,
                "dropoff_y": passenger.dropoff_y,
                "payout": passenger.payout,
                "picked_up": passenger.picked_up,
                "completed": passenger.completed,
            }
        )

    return snapshot


def choose_step_toward(current: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    cx, cy = current
    tx, ty = target

    candidates: list[tuple[int, int]] = []

    if tx < cx:
        candidates.append((-1, 0))
    elif tx > cx:
        candidates.append((1, 0))

    if ty < cy:
        candidates.append((0, -1))
    elif ty > cy:
        candidates.append((0, 1))

    if not candidates:
        return random.choice([(0, -1), (-1, 0), (0, 1), (1, 0)])

    return random.choice(candidates)


def choose_best_value_target(manager: TaxiManager) -> tuple[tuple[int, int], dict[str, Any]]:
    taxi_pos = manager.taxi.position()

    if manager.taxi.current_passenger is not None:
        passenger = manager.taxi.current_passenger
        target = passenger.dropoff_position()

        meta = {
            "decision_mode": "dropoff",
            "target_passenger_id": find_passenger_id(manager, passenger),
            "target_x": target[0],
            "target_y": target[1],
            "pickup_distance": 0,
            "trip_distance": manhattan(taxi_pos, target),
            "score": None,
            "payout": passenger.payout,
        }
        return target, meta

    candidates = []

    for i, passenger in enumerate(manager.passengers):
        if passenger.completed or passenger.picked_up:
            continue

        pickup = passenger.pickup_position()
        dropoff = passenger.dropoff_position()
        payout = passenger.payout

        pickup_distance = manhattan(taxi_pos, pickup)
        trip_distance = manhattan(pickup, dropoff)

        score = payout - ALPHA_PICKUP * pickup_distance - BETA_TRIP * trip_distance

        candidates.append(
            {
                "passenger": passenger,
                "passenger_id": get_passenger_id(i),
                "pickup": pickup,
                "dropoff": dropoff,
                "payout": payout,
                "pickup_distance": pickup_distance,
                "trip_distance": trip_distance,
                "score": score,
            }
        )

    if not candidates:
        random_target = taxi_pos
        meta = {
            "decision_mode": "fallback",
            "target_passenger_id": None,
            "target_x": random_target[0],
            "target_y": random_target[1],
            "pickup_distance": 0,
            "trip_distance": 0,
            "score": None,
            "payout": 0.0,
        }
        return random_target, meta

    best = max(candidates, key=lambda x: x["score"])

    meta = {
        "decision_mode": "pickup",
        "target_passenger_id": best["passenger_id"],
        "target_x": best["pickup"][0],
        "target_y": best["pickup"][1],
        "pickup_distance": best["pickup_distance"],
        "trip_distance": best["trip_distance"],
        "score": best["score"],
        "payout": best["payout"],
    }

    return best["pickup"], meta


def find_passenger_id(manager: TaxiManager, passenger_obj: Any) -> str | None:
    for i, passenger in enumerate(manager.passengers):
        if passenger is passenger_obj:
            return get_passenger_id(i)
    return None


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


def create_manager(seed: int) -> TaxiManager:
    random.seed(seed)
    manager = TaxiManager(width=WIDTH, height=HEIGHT, num_passengers=NUM_PASSENGERS)
    manager.create_passengers()
    return manager


def flush_transition_chunk(
    transitions_chunk: list[dict[str, Any]],
    output_file,
) -> None:
    if not transitions_chunk:
        return

    pickle.dump(transitions_chunk, output_file)
    transitions_chunk.clear()


# ============================================================
# MAIN LOG GENERATION
# ============================================================

def generate_logs() -> None:
    ensure_data_dir()

    sim_time = datetime(2026, 1, 1, 8, 0, 0)

    total_events = 0
    total_transitions = 0
    transitions_chunk: list[dict[str, Any]] = []

    with RAW_LOG_FILE.open("w", encoding="utf-8") as raw_f, TRANSITIONS_FILE.open("wb") as trans_f:
        for episode_id in range(NUM_EPISODES):
            manager = create_manager(BASE_SEED + episode_id)

            step_idx = 0
            done = manager.is_done()

            while not done and step_idx < MAX_STEPS_PER_EPISODE:
                state = build_state(manager)

                taxi_pos_before = manager.taxi.position()
                carrying_before = find_passenger_id(manager, manager.taxi.current_passenger)

                target_pos, heuristic_meta = choose_best_value_target(manager)
                dx, dy = choose_step_toward(taxi_pos_before, target_pos)

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
                    "policy_name": "best_value_v1",

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
