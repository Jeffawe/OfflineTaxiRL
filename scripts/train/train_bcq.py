import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import pickle

import numpy as np
import torch
from d3rlpy.algos import DiscreteBCQConfig
from d3rlpy.dataset import MDPDataset


QUALITY_TO_TRANSITIONS_FILE = {
    "expert": Path("data/transitions_best_value.pkl"),
    "mild": Path("data/transitions_mild_noisy.pkl"),
    "mixed": Path("data/transitions_mixed_quality.pkl"),
    "poor": Path("data/transitions_poor_noisy.pkl"),
}

QUALITY_TO_MODEL_FILE = {
    "expert": Path("data/offline_rl_bcq_expert.d3"),
    "mild": Path("data/offline_rl_bcq_mild.d3"),
    "mixed": Path("data/offline_rl_bcq_mixed.d3"),
    "poor": Path("data/offline_rl_bcq_poor.d3"),
}

N_STEPS = 20_000
N_STEPS_PER_EPOCH = 1_000
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
GAMMA = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a discrete BCQ model.")
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_TO_TRANSITIONS_FILE),
        default="expert",
        help="Logged data quality level to train on.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> MDPDataset:
    with path.open("rb") as f:
        transitions = pickle.load(f)

    observations = np.asarray([t["state"] for t in transitions], dtype=np.float32)
    actions = np.asarray([t["action"] for t in transitions], dtype=np.int64)
    rewards = np.asarray([t["reward"] for t in transitions], dtype=np.float32)
    terminals = np.asarray([float(t["done"]) for t in transitions], dtype=np.float32)

    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        action_size=4,
    )


def main() -> None:
    args = parse_args()
    transitions_file = QUALITY_TO_TRANSITIONS_FILE[args.quality]
    model_file = QUALITY_TO_MODEL_FILE[args.quality]

    print(f"Training BCQ on quality='{args.quality}' using {transitions_file}")
    dataset = load_dataset(transitions_file)
    device = "cuda:0" if torch.cuda.is_available() else False

    algo = DiscreteBCQConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
    ).create(device=device)

    algo.fit(
        dataset,
        n_steps=N_STEPS,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        experiment_name=f"offline_rl_taxi_bcq_{args.quality}",
        show_progress=True,
    )

    model_file.parent.mkdir(parents=True, exist_ok=True)
    algo.save(str(model_file))
    print(f"Saved offline RL BCQ model to {model_file}")


if __name__ == "__main__":
    main()
