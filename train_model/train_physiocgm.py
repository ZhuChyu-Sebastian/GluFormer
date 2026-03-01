import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_model.modeling import CrossModalGluFormer


DEFAULT_SENSOR_COLUMNS = [
    "E4_HR",
    "EDA",
    "TEMP",
    "BVP",
    "E4_Acc_x_x",
    "E4_Acc_y_x",
    "E4_Acc_z_x",
    "Accel_Vertical",
    "Accel_Lateral",
    "Accel_Sagittal",
    "BreathingWaveform",
]


class MultimodalWindowDataset(Dataset):
    def __init__(self, glucose_windows: list[torch.Tensor], sensor_windows: list[torch.Tensor]):
        self.glucose_windows = glucose_windows
        self.sensor_windows = sensor_windows

    def __len__(self) -> int:
        return len(self.glucose_windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.glucose_windows[idx], self.sensor_windows[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_glucose(values: pd.Series, glucose_min: int, glucose_max: int) -> np.ndarray:
    values = values.astype(float).clip(glucose_min, glucose_max).round().astype(int)
    return (values - glucose_min).to_numpy(dtype=np.int64)


def parse_sensor_columns(sensor_columns_arg: str) -> list[str]:
    cols = [c.strip() for c in sensor_columns_arg.split(",") if c.strip()]
    if not cols:
        raise ValueError("--sensor-columns 不能为空")
    return cols


def load_sequences(
    dataset_dir: Path,
    glucose_column: str,
    time_column: str | None,
    sensor_columns: list[str],
    glucose_min: int,
    glucose_max: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    glucose_sequences: list[np.ndarray] = []
    sensor_sequences_raw: list[np.ndarray] = []
    sensor_rows_for_stats: list[np.ndarray] = []

    for csv_path in sorted(dataset_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if glucose_column not in df.columns:
            raise ValueError(f"{csv_path} 缺少列: {glucose_column}")

        missing_sensor_cols = [col for col in sensor_columns if col not in df.columns]
        if missing_sensor_cols:
            raise ValueError(f"{csv_path} 缺少传感器列: {missing_sensor_cols}")

        if time_column and time_column in df.columns:
            df = df.sort_values(time_column)

        mask = df[glucose_column].notna()
        df = df.loc[mask, [glucose_column] + sensor_columns].copy()
        if df.empty:
            continue

        glucose_tokens = tokenize_glucose(df[glucose_column], glucose_min, glucose_max)

        sensor_df = df[sensor_columns].astype(float)
        sensor_df = sensor_df.ffill().bfill().fillna(0.0)
        sensor_values = sensor_df.to_numpy(dtype=np.float32)

        glucose_sequences.append(glucose_tokens)
        sensor_sequences_raw.append(sensor_values)
        sensor_rows_for_stats.append(sensor_values)

    if not glucose_sequences:
        raise ValueError("没有有效序列，请检查 dataset/*.csv 和列名配置")

    sensor_all = np.concatenate(sensor_rows_for_stats, axis=0)
    sensor_mean = sensor_all.mean(axis=0)
    sensor_std = sensor_all.std(axis=0)
    sensor_std = np.where(sensor_std < 1e-6, 1.0, sensor_std)

    sensor_sequences = [((seq - sensor_mean) / sensor_std).astype(np.float32) for seq in sensor_sequences_raw]
    return glucose_sequences, sensor_sequences, sensor_mean.astype(np.float32), sensor_std.astype(np.float32)


def build_windows(
    glucose_sequences: list[np.ndarray],
    sensor_sequences: list[np.ndarray],
    seq_len: int,
    stride: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    glucose_windows: list[torch.Tensor] = []
    sensor_windows: list[torch.Tensor] = []
    window_len = seq_len + 1

    for glucose_seq, sensor_seq in zip(glucose_sequences, sensor_sequences):
        if len(glucose_seq) < window_len:
            continue
        for start in range(0, len(glucose_seq) - window_len + 1, stride):
            end = start + window_len
            glucose_windows.append(torch.tensor(glucose_seq[start:end], dtype=torch.long))
            sensor_windows.append(torch.tensor(sensor_seq[start:end], dtype=torch.float32))

    if not glucose_windows:
        raise ValueError("没有构建出任何训练窗口，请降低 --seq-len 或检查数据长度")
    return glucose_windows, sensor_windows


def evaluate(model: CrossModalGluFormer, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for glucose_batch, sensor_batch in dataloader:
            glucose_batch = glucose_batch.to(device)
            sensor_batch = sensor_batch.to(device)

            inputs = glucose_batch[:, :-1]
            targets = glucose_batch[:, 1:]
            sensor_inputs = sensor_batch[:, :-1, :]

            logits = model(inputs, sensor_inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())
    return float(np.mean(losses)), float(np.mean(accs))


def main() -> None:
    parser = argparse.ArgumentParser(description="在 dataset/PhyscioCGM CSV 上进行双塔/跨模态注意力融合训练")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("train_model/physiocgm_gluformer_multimodal.pt"))
    parser.add_argument("--glucose-column", type=str, default="Glucose")
    parser.add_argument("--time-column", type=str, default="Time")
    parser.add_argument("--sensor-columns", type=str, default=",".join(DEFAULT_SENSOR_COLUMNS))
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
    sensor_columns = parse_sensor_columns(args.sensor_columns)

    glucose_sequences, sensor_sequences, sensor_mean, sensor_std = load_sequences(
        dataset_dir=args.dataset_dir,
        glucose_column=args.glucose_column,
        time_column=args.time_column,
        sensor_columns=sensor_columns,
        glucose_min=args.glucose_min,
        glucose_max=args.glucose_max,
    )
    glucose_windows, sensor_windows = build_windows(
        glucose_sequences=glucose_sequences,
        sensor_sequences=sensor_sequences,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    indices = list(range(len(glucose_windows)))
    random.shuffle(indices)
    split_idx = int(len(indices) * args.train_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_ds = MultimodalWindowDataset(
        [glucose_windows[i] for i in train_idx],
        [sensor_windows[i] for i in train_idx],
    )
    val_ds = MultimodalWindowDataset(
        [glucose_windows[i] for i in val_idx],
        [sensor_windows[i] for i in val_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    vocab_size = args.glucose_max - args.glucose_min + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalGluFormer(
        vocab_size=vocab_size,
        sensor_dim=len(sensor_columns),
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

        for glucose_batch, sensor_batch in train_loader:
            glucose_batch = glucose_batch.to(device)
            sensor_batch = sensor_batch.to(device)

            inputs = glucose_batch[:, :-1]
            targets = glucose_batch[:, 1:]
            sensor_inputs = sensor_batch[:, :-1, :]

            logits = model(inputs, sensor_inputs)
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
                "sensor_dim": len(sensor_columns),
                "sensor_columns": sensor_columns,
                "sensor_mean": sensor_mean.tolist(),
                "sensor_std": sensor_std.tolist(),
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
