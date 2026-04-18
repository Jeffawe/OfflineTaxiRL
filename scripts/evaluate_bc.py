import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from environment.taxiManager import TaxiManager
from models.bc_model import BehaviorCloningModel


MODEL_FILE = "data/bc_model.pt"

WIDTH = 10
HEIGHT = 10
NUM_EPISODES = 20

MOVE_PENALTY = -0.1
INVALID_MOVE_PENALTY = -1.0
PICKUP_BONUS = 0.0

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


def load_model(device: torch.device) -> BehaviorCloningModel:
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model = BehaviorCloningModel(
        input_dim=checkpoint["input_dim"],
        num_actions=checkpoint["num_actions"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

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

def run_episode(model: BehaviorCloningModel, device: torch.device) -> dict[str, float | int | bool]:
    manager = TaxiManager(width=WIDTH, height=HEIGHT, num_passengers=2)
    manager.create_passengers()

    total_reward = 0.0
    max_steps = 100
    step_count = 0

    invalid_moves = 0
    pickup_count = 0
    dropoff_count = 0

    while not manager.is_done() and step_count < max_steps:
        state = build_state(manager)
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(state_tensor)
            action_id = int(torch.argmax(logits, dim=1).item())

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    num_episodes = NUM_EPISODES
    results = []

    for _ in range(num_episodes):
        results.append(run_episode(model, device))

    average_reward = sum(r["reward"] for r in results) / num_episodes
    success_rate = sum(1 for r in results if r["success"]) / num_episodes
    average_episode_length = sum(r["episode_length"] for r in results) / num_episodes
    pickup_rate = sum(1 for r in results if r["picked_up_any"]) / num_episodes
    dropoff_rate = sum(1 for r in results if r["dropped_off_any"]) / num_episodes
    average_invalid_move_rate = sum(r["invalid_move_rate"] for r in results) / num_episodes

    average_pickups_per_episode = sum(r["pickup_count"] for r in results) / num_episodes
    average_dropoffs_per_episode = sum(r["dropoff_count"] for r in results) / num_episodes
    average_invalid_moves_per_episode = sum(r["invalid_moves"] for r in results) / num_episodes

    print(f"Rollout evaluation over {num_episodes} episodes")
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