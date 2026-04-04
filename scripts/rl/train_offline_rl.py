import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle

import numpy as np
import torch
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset


TRANSITIONS_FILE = Path("data/transitions_best_value.pkl")
MODEL_FILE = Path("data/offline_rl_cql.d3")

N_STEPS = 20_000
N_STEPS_PER_EPOCH = 1_000
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
GAMMA = 0.99


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
    dataset = load_dataset(TRANSITIONS_FILE)
    device = "cuda:0" if torch.cuda.is_available() else False

    algo = DiscreteCQLConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
    ).create(device=device)

    algo.fit(
        dataset,
        n_steps=N_STEPS,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        experiment_name="offline_rl_taxi_cql",
        show_progress=True,
    )

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    algo.save(str(MODEL_FILE))
    print(f"Saved offline RL model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
