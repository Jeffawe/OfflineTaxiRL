import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.bc_model import BehaviorCloningModel


TRANSITIONS_FILE = Path("data/transitions_best_value.pkl")
MODEL_FILE = Path("data/bc_model.pt")

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.8
SEED = 42
EARLY_STOPPING_PATIENCE = 5


class BCDataset(Dataset):
    def __init__(self, states: list[list[float]], actions: list[int]) -> None:
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index]


def load_bc_data(path: Path) -> tuple[list[list[float]], list[int]]:
    with path.open("rb") as f:
        transitions = pickle.load(f)

    states: list[list[float]] = []
    actions: list[int] = []

    for t in transitions:
        states.append(t["state"])
        actions.append(int(t["action"]))

    return states, actions


def split_data(
    states: list[list[float]],
    actions: list[int],
    train_split: float = TRAIN_SPLIT,
) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    indices = list(range(len(states)))
    random.shuffle(indices)

    split_index = int(len(indices) * train_split)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    train_states = [states[i] for i in train_indices]
    train_actions = [actions[i] for i in train_indices]

    val_states = [states[i] for i in val_indices]
    val_actions = [actions[i] for i in val_indices]

    return train_states, train_actions, val_states, val_actions


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def top_k_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 2,
) -> float:
    k = min(k, logits.size(1))
    topk_indices = torch.topk(logits, k=k, dim=1).indices
    correct = (topk_indices == targets.unsqueeze(1)).any(dim=1).float().sum().item()
    total = targets.size(0)
    return correct / total


def update_confusion_matrix(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    predictions = torch.argmax(logits, dim=1)

    for true_label, predicted_label in zip(targets.view(-1), predictions.view(-1)):
        confusion[true_label.long(), predicted_label.long()] += 1


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_actions: int,
) -> tuple[float, float, float, torch.Tensor]:
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_top2_acc = 0.0
    num_batches = 0

    confusion = torch.zeros((num_actions, num_actions), dtype=torch.int64)

    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)

            logits = model(states)
            loss = criterion(logits, actions)

            acc = accuracy_from_logits(logits, actions)
            top2_acc = top_k_accuracy_from_logits(logits, actions, k=2)

            total_loss += loss.item()
            total_acc += acc
            total_top2_acc += top2_acc
            num_batches += 1

            update_confusion_matrix(confusion, logits.cpu(), actions.cpu())

    if num_batches == 0:
        return 0.0, 0.0, 0.0, confusion

    return (
        total_loss / num_batches,
        total_acc / num_batches,
        total_top2_acc / num_batches,
        confusion,
    )


def print_confusion_matrix(confusion: torch.Tensor) -> None:
    print("\nConfusion Matrix (rows=true action, cols=predicted action):")
    print(confusion)

    print("\nPer-class accuracy:")
    for i in range(confusion.size(0)):
        row_total = confusion[i].sum().item()
        correct = confusion[i, i].item()
        class_acc = (correct / row_total) if row_total > 0 else 0.0
        print(f"Action {i}: {class_acc:.4f} ({correct}/{row_total})")


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    states, actions = load_bc_data(TRANSITIONS_FILE)

    if not states:
        raise ValueError("No states found in transition file.")

    input_dim = len(states[0])
    num_actions = len(set(actions))

    train_states, train_actions, val_states, val_actions = split_data(states, actions)

    train_dataset = BCDataset(train_states, train_actions)
    val_dataset = BCDataset(val_states, val_actions)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BehaviorCloningModel(input_dim=input_dim, num_actions=num_actions).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0.0
        total_train_acc = 0.0
        num_train_batches = 0

        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()

            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()

            acc = accuracy_from_logits(logits, batch_actions)

            total_train_loss += loss.item()
            total_train_acc += acc
            num_train_batches += 1

        train_loss = total_train_loss / num_train_batches
        train_acc = total_train_acc / num_train_batches

        val_loss, val_acc, val_top2_acc, _ = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_actions=num_actions,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top2_acc={val_top2_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"after {EARLY_STOPPING_PATIENCE} epochs without val_loss improvement."
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_val_loss, final_val_acc, final_val_top2_acc, final_confusion = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        num_actions=num_actions,
    )

    print("\nBest model validation metrics:")
    print(f"val_loss={final_val_loss:.4f}")
    print(f"val_acc={final_val_acc:.4f}")
    print(f"val_top2_acc={final_val_top2_acc:.4f}")

    print_confusion_matrix(final_confusion)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_actions": num_actions,
        },
        MODEL_FILE,
    )

    print(f"\nSaved BC model to {MODEL_FILE}")


if __name__ == "__main__":
    main()