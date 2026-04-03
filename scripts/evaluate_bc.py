import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from environment.taxiManager import TaxiManager
from models.bc_model import BehaviorCloningModel


MODEL_FILE = "data/bc_model.pt"

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


def run_episode(model: BehaviorCloningModel, device: torch.device) -> float:
    manager = TaxiManager(width=20, height=20, num_passengers=2)
    manager.create_passengers()

    total_reward = 0.0
    max_steps = 100
    step_count = 0

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

        if moved:
            picked_up = manager.pickup_passenger()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    rewards = []
    num_episodes = 20

    for _ in range(num_episodes):
        rewards.append(run_episode(model, device))

    average_reward = sum(rewards) / len(rewards)
    print(f"Average reward over {num_episodes} episodes: {average_reward:.2f}")


if __name__ == "__main__":
    main()
