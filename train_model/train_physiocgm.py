import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_model.modeling import TransformerModel


class GlucoseWindowDataset(Dataset):
    def __init__(self, windows: list[torch.Tensor]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_glucose(values: pd.Series, glucose_min: int, glucose_max: int) -> np.ndarray:
    values = values.astype(float).clip(glucose_min, glucose_max).round().astype(int)
    return (values - glucose_min).to_numpy(dtype=np.int64)


def build_windows(tokens: np.ndarray, seq_len: int, stride: int) -> list[torch.Tensor]:
    window_len = seq_len + 1
    if len(tokens) < window_len:
        return []
    windows = []
    for start in range(0, len(tokens) - window_len + 1, stride):
        windows.append(torch.tensor(tokens[start : start + window_len], dtype=torch.long))
    return windows


def load_windows(dataset_dir: Path, seq_len: int, stride: int, glucose_column: str, time_column: str | None,
                 glucose_min: int, glucose_max: int) -> list[torch.Tensor]:
    all_windows = []
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if glucose_column not in df.columns:
            raise ValueError(f"{csv_path} 缺少列: {glucose_column}")

        if time_column and time_column in df.columns:
            df = df.sort_values(time_column)

        glucose = df[glucose_column].dropna()
        tokens = tokenize_glucose(glucose, glucose_min, glucose_max)
        all_windows.extend(build_windows(tokens, seq_len=seq_len, stride=stride))

    if not all_windows:
        raise ValueError("没有构建出任何训练窗口，请检查数据长度和 seq-len。")
    return all_windows


def evaluate(model: TransformerModel, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())
    return float(np.mean(losses)), float(np.mean(accs))


def main() -> None:
    parser = argparse.ArgumentParser(description="在 dataset/PhyscioCGM CSV 上训练 GluFormer")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("train_model/physiocgm_gluformer.pt"))
    parser.add_argument("--glucose-column", type=str, default="Glucose")
    parser.add_argument("--time-column", type=str, default="Time")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--glucose-min", type=int, default=40)
    parser.add_argument("--glucose-max", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)

    windows = load_windows(
        dataset_dir=args.dataset_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        glucose_column=args.glucose_column,
        time_column=args.time_column,
        glucose_min=args.glucose_min,
        glucose_max=args.glucose_max,
    )
    random.shuffle(windows)
    split_idx = int(len(windows) * args.train_ratio)
    train_ds = GlucoseWindowDataset(windows[:split_idx])
    val_ds = GlucoseWindowDataset(windows[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    vocab_size = args.glucose_max - args.glucose_min + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_length=args.seq_len,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | train_loss={np.mean(train_losses):.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "n_embd": args.n_embd,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "max_seq_length": args.seq_len,
                "dropout": args.dropout,
                "dim_feedforward": args.dim_feedforward,
                "glucose_min": args.glucose_min,
                "glucose_max": args.glucose_max,
            },
        },
        args.output,
    )
    print(f"模型已保存: {args.output}")


if __name__ == "__main__":
    main()
