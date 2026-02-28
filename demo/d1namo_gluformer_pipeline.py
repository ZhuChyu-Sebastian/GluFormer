import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_FEATURES = ["ppg", "imu", "temperature", "meal"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultimodalGluFormer(nn.Module):
    def __init__(self, vocab_size: int, feature_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens) + self.feature_proj(features)
        x = self.pos_enc(x)
        seq_len = tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()
        hidden = self.encoder(x, mask=causal_mask)
        return self.head(hidden)


@dataclass
class PreparedData:
    train_tokens: np.ndarray
    train_features: np.ndarray
    val_tokens: np.ndarray
    val_features: np.ndarray
    test_tokens: np.ndarray
    test_features: np.ndarray
    test_times: np.ndarray


def tokenize_glucose(values: np.ndarray, g_min: float = 40.0, g_max: float = 500.0, bins: int = 460) -> np.ndarray:
    values = np.clip(values, g_min, g_max)
    scaled = np.floor(values - g_min).astype(np.int64)
    return np.clip(scaled, 0, bins - 1)


def build_windows(df: pd.DataFrame, feature_cols: List[str], seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens_all, feats_all, times_all = [], [], []
    for _, g in df.groupby("participant_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        total_len = seq_len + pred_len
        if len(g) < total_len:
            continue
        tok = g["cgm_token"].to_numpy(np.int64)
        feat = g[feature_cols].to_numpy(np.float32)
        ts = g["timestamp"].astype(str).to_numpy()
        for i in range(len(g) - total_len + 1):
            tokens_all.append(tok[i : i + total_len])
            feats_all.append(feat[i : i + total_len])
            times_all.append(ts[i : i + total_len])
    return np.array(tokens_all), np.array(feats_all), np.array(times_all)


def prepare_dataset(input_csv: Path, output_dir: Path, feature_cols: List[str], seq_len: int, pred_len: int) -> PreparedData:
    df = pd.read_csv(input_csv)
    need = {"participant_id", "timestamp", "cgm"} | set(feature_cols)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV缺少必要列: {miss}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["participant_id", "timestamp"]).copy()

    for col in feature_cols:
        df[col] = df[col].astype(float).fillna(df[col].median())
        mu, sigma = df[col].mean(), df[col].std() + 1e-6
        df[col] = (df[col] - mu) / sigma

    df["cgm_token"] = tokenize_glucose(df["cgm"].to_numpy())
    tokens, feats, times = build_windows(df, feature_cols, seq_len, pred_len)
    if len(tokens) == 0:
        raise ValueError("未构建出可用窗口，请检查采样频率或增大数据量。")

    n = len(tokens)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    idx1, idx2 = n_train, n_train + n_val

    prepared = PreparedData(
        train_tokens=tokens[:idx1],
        train_features=feats[:idx1],
        val_tokens=tokens[idx1:idx2],
        val_features=feats[idx1:idx2],
        test_tokens=tokens[idx2:],
        test_features=feats[idx2:],
        test_times=times[idx2:],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "d1namo_prepared.npz",
        **prepared.__dict__,
        feature_cols=np.array(feature_cols),
        seq_len=seq_len,
        pred_len=pred_len,
    )
    return prepared


def run_epoch(model, tokens, features, optimizer, device):
    model.train()
    losses = []
    batch_size = 32
    for i in range(0, len(tokens), batch_size):
        bt = torch.tensor(tokens[i : i + batch_size], dtype=torch.long, device=device)
        bf = torch.tensor(features[i : i + batch_size], dtype=torch.float32, device=device)
        x_t, y_t = bt[:, :-1], bt[:, 1:]
        x_f = bf[:, :-1]
        logits = model(x_t, x_f)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_t.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, tokens, features, device):
    model.eval()
    batch_size = 64
    losses = []
    for i in range(0, len(tokens), batch_size):
        bt = torch.tensor(tokens[i : i + batch_size], dtype=torch.long, device=device)
        bf = torch.tensor(features[i : i + batch_size], dtype=torch.float32, device=device)
        logits = model(bt[:, :-1], bf[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), bt[:, 1:].reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def forecast(model, tokens, features, seq_len: int, pred_len: int, device):
    model.eval()
    preds = []
    for i in range(len(tokens)):
        t = torch.tensor(tokens[i : i + 1, :seq_len], dtype=torch.long, device=device)
        f = torch.tensor(features[i : i + 1, :seq_len], dtype=torch.float32, device=device)
        out = []
        for step in range(pred_len):
            logits = model(t, f)[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            out.append(nxt.item())
            t = torch.cat([t, nxt], dim=1)
            next_feat = torch.tensor(features[i : i + 1, seq_len + step : seq_len + step + 1], dtype=torch.float32, device=device)
            f = torch.cat([f, next_feat], dim=1)
        preds.append(out)
    return np.array(preds)


def detokenize(tokens: np.ndarray, offset: float = 40.0) -> np.ndarray:
    return tokens.astype(float) + offset


def save_plots(times, y_true, y_pred, out_dir: Path, max_figures: int = 5):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(max_figures, len(y_true))):
        plt.figure(figsize=(10, 4))
        plt.plot(times[i], y_true[i], label="真实CGM", linewidth=2)
        plt.plot(times[i], y_pred[i], label="预测CGM", linestyle="--")
        plt.xticks(rotation=35)
        plt.ylabel("mg/dL")
        plt.title(f"D1NAMO 预测示例 #{i}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"forecast_{i}.png", dpi=140)
        plt.close()


def generate_synthetic_csv(path: Path, n_participants: int = 10, points_per_p: int = 350):
    rng = np.random.default_rng(42)
    rows = []
    for pid in range(n_participants):
        t0 = pd.Timestamp("2024-01-01")
        base = 95 + 5 * rng.normal()
        for k in range(points_per_p):
            timestamp = t0 + pd.Timedelta(minutes=5 * k)
            meal = 1.0 if k % 65 in [0, 1, 2] else 0.0
            ppg = np.sin(k / 12) + 0.1 * rng.normal()
            imu = np.cos(k / 20) + 0.2 * rng.normal()
            temp = 36.5 + 0.2 * np.sin(k / 18) + 0.05 * rng.normal()
            cgm = base + 25 * meal + 8 * ppg - 5 * imu + 4 * (temp - 36.5) + 4 * rng.normal()
            rows.append([f"P{pid:03d}", timestamp, cgm, ppg, imu, temp, meal])
    pd.DataFrame(rows, columns=["participant_id", "timestamp", "cgm", "ppg", "imu", "temperature", "meal"]).to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description="GluFormer 在 D1NAMO 多模态数据上的训练/预测/可视化流程")
    p.add_argument("--mode", choices=["prepare", "train", "predict", "visualize", "all", "demo"], default="all")
    p.add_argument("--input_csv", type=Path, default=Path("data/d1namo_merged.csv"))
    p.add_argument("--workdir", type=Path, default=Path("outputs/d1namo"))
    p.add_argument("--feature_cols", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--seq_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    if args.mode == "demo":
        args.input_csv.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_csv(args.input_csv)
        args.mode = "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared_path = args.workdir / "d1namo_prepared.npz"

    if args.mode in {"prepare", "all"}:
        prepare_dataset(args.input_csv, args.workdir, args.feature_cols, args.seq_len, args.pred_len)

    if args.mode in {"train", "predict", "visualize", "all"}:
        blob = np.load(prepared_path, allow_pickle=True)
        train_tokens = blob["train_tokens"]
        train_features = blob["train_features"]
        val_tokens = blob["val_tokens"]
        val_features = blob["val_features"]

    model_path = args.workdir / "d1namo_gluformer.pt"
    if args.mode in {"train", "all"}:
        model = MultimodalGluFormer(
            vocab_size=460,
            feature_dim=len(args.feature_cols),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            max_len=args.seq_len + args.pred_len + 4,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        hist = []
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_tokens, train_features, opt, device)
            va = evaluate(model, val_tokens, val_features, device)
            hist.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
            print(f"epoch={epoch} train={tr:.4f} val={va:.4f}")
        torch.save(model.state_dict(), model_path)
        pd.DataFrame(hist).to_csv(args.workdir / "train_log.csv", index=False)

    if args.mode in {"predict", "visualize", "all"}:
        blob = np.load(prepared_path, allow_pickle=True)
        test_tokens = blob["test_tokens"]
        test_features = blob["test_features"]
        test_times = blob["test_times"]

        model = MultimodalGluFormer(
            vocab_size=460,
            feature_dim=len(args.feature_cols),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            max_len=args.seq_len + args.pred_len + 4,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        pred_tokens = forecast(model, test_tokens, test_features, args.seq_len, args.pred_len, device)
        true_tokens = test_tokens[:, args.seq_len : args.seq_len + args.pred_len]
        y_true = detokenize(true_tokens)
        y_pred = detokenize(pred_tokens)
        times = test_times[:, args.seq_len : args.seq_len + args.pred_len]

        mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
        rmse = np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
        metrics = {"mae": float(mae), "rmse": float(rmse), "n_test_windows": int(len(test_tokens))}

        pd.DataFrame({
            "timestamp": times.reshape(-1),
            "y_true": y_true.reshape(-1),
            "y_pred": y_pred.reshape(-1),
        }).to_csv(args.workdir / "predictions.csv", index=False)

        with open(args.workdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print("metrics:", metrics)

        if args.mode in {"visualize", "all"}:
            save_plots(times, y_true, y_pred, args.workdir / "figures")


if __name__ == "__main__":
    main()
