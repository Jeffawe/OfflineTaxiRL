import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random
import statistics
import torch

from environment.taxiManager import TaxiManager
from models.bc_model import BehaviorCloningModel


DATA_DIR = Path("data")
QUALITY_TO_MODEL_FILE = {
    "expert": DATA_DIR / "bc_model_expert.pt",
    "mild": DATA_DIR / "bc_model_mild.pt",
    "mixed": DATA_DIR / "bc_model_mixed.pt",
    "poor": DATA_DIR / "bc_model_poor.pt",
}
LEGACY_MODEL_FILE = DATA_DIR / "bc_model.pt"
BC_EPOCH_OPTIONS = [40, 60, 100]

WIDTH = 15
HEIGHT = 15
NUM_EPISODES = 100
EVAL_SEEDS = [0, 1, 2, 3, 4]
MAX_STEPS = 400

MOVE_PENALTY = -0.1
INVALID_MOVE_PENALTY = -1.0
PICKUP_BONUS = 1.0

ID_TO_ACTION = {
    0: (0, -1),   # up
    1: (-1, 0),   # left
    2: (0, 1),    # down
    3: (1, 0),    # right
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a behavior cloning model.")
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_TO_MODEL_FILE),
        default=None,
        help="Evaluate the BC model trained on this data quality. If omitted, use the first BC model found in data.",
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Evaluate all configured epoch-budget BC checkpoints for the selected quality.",
    )
    return parser.parse_args()


def find_first_bc_model() -> Path:
    candidates = sorted(DATA_DIR.glob("bc_model_*.pt"))

    if candidates:
        return candidates[0]

    if LEGACY_MODEL_FILE.exists():
        return LEGACY_MODEL_FILE

    raise FileNotFoundError(
        "No BC model found. Expected one of data/bc_model_*.pt "
        f"or legacy file {LEGACY_MODEL_FILE}."
    )


def resolve_model_file(quality: str | None) -> Path:
    if quality is None:
        return find_first_bc_model()

    model_file = QUALITY_TO_MODEL_FILE[quality]
    if not model_file.exists():
        raise FileNotFoundError(
            f"BC model for quality='{quality}' not found: {model_file}"
        )

    return model_file


def model_file_for_epochs(base_model_file: Path, epochs: int) -> Path:
    return base_model_file.with_name(f"{base_model_file.stem}_e{epochs}{base_model_file.suffix}")


def resolve_model_files(quality: str | None, multiple: bool) -> list[Path]:
    if not multiple:
        return [resolve_model_file(quality)]

    if quality is None:
        raise ValueError("--multiple requires --quality so the evaluator knows which BC sweep to load.")

    base_model_file = QUALITY_TO_MODEL_FILE[quality]
    model_files = [model_file_for_epochs(base_model_file, epochs) for epochs in BC_EPOCH_OPTIONS]
    missing_files = [str(path) for path in model_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Missing BC epoch-sweep model files: " + ", ".join(missing_files)
        )

    return model_files


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


def load_model(model_file: Path, device: torch.device) -> BehaviorCloningModel:
    checkpoint = torch.load(model_file, map_location=device)
    model = BehaviorCloningModel(
        input_dim=checkpoint["input_dim"],
        num_actions=checkpoint["num_actions"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_checkpoint(model_file: Path, device: torch.device) -> tuple[BehaviorCloningModel, dict]:
    checkpoint = torch.load(model_file, map_location=device)
    model = BehaviorCloningModel(
        input_dim=checkpoint["input_dim"],
        num_actions=checkpoint["num_actions"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint

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
    max_steps = MAX_STEPS
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


def evaluate_seed(
    model: BehaviorCloningModel,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    random.seed(seed)
    results = [run_episode(model, device) for _ in range(NUM_EPISODES)]

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
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_files = resolve_model_files(args.quality, args.multiple)

    for model_file in model_files:
        model, checkpoint = load_checkpoint(model_file, device)
        summaries = [evaluate_seed(model, device, seed) for seed in EVAL_SEEDS]
        success_rates = [summary["success_rate"] for summary in summaries]
        epochs = checkpoint.get("epochs")

        print(f"Model file: {model_file}")
        if epochs is not None:
            print(f"Epoch budget: {epochs}")
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
        print()


if __name__ == "__main__":
    main()
