import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_model.modeling import CrossModalGluFormer


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        probs = torch.softmax(logits, dim=-1)
    else:
        values, _ = torch.topk(logits, k=min(k, logits.size(-1)))
        threshold = values[..., -1, None]
        masked = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(masked, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用双塔/跨模态注意力 GluFormer 预测未来 CGM token")
    parser.add_argument("--checkpoint", type=Path, default=Path("train_model/physiocgm_gluformer_multimodal.pt"))
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/0.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("model_usage/physiocgm_predictions.csv"))
    parser.add_argument("--glucose-column", type=str, default="Glucose")
    parser.add_argument("--time-column", type=str, default="Time")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--predict-steps", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    model = CrossModalGluFormer(
        vocab_size=config["vocab_size"],
        sensor_dim=config["sensor_dim"],
        n_embd=config["n_embd"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_length=config["max_seq_length"],
        dropout=config["dropout"],
        dim_feedforward=config["dim_feedforward"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    df = pd.read_csv(args.input_csv)
    if args.time_column in df.columns:
        df = df.sort_values(args.time_column)

    sensor_cols = config["sensor_columns"]
    missing_sensor_cols = [col for col in sensor_cols if col not in df.columns]
    if missing_sensor_cols:
        raise ValueError(f"输入 CSV 缺少传感器列: {missing_sensor_cols}")

    glucose_mask = df[args.glucose_column].notna()
    df = df.loc[glucose_mask, [args.glucose_column] + sensor_cols].copy()

    glucose_min = config["glucose_min"]
    glucose_max = config["glucose_max"]
    values = (
        df[args.glucose_column]
        .dropna()
        .astype(float)
        .clip(glucose_min, glucose_max)
        .round()
        .astype(int)
    )
    glucose_tokens = torch.tensor((values - glucose_min).to_numpy(), dtype=torch.long).unsqueeze(0)

    sensor_df = df[sensor_cols].astype(float).ffill().bfill().fillna(0.0)
    sensor_values = sensor_df.to_numpy(dtype=np.float32)
    sensor_mean = np.array(config["sensor_mean"], dtype=np.float32)
    sensor_std = np.array(config["sensor_std"], dtype=np.float32)
    sensor_values = (sensor_values - sensor_mean) / sensor_std
    sensor_features = torch.tensor(sensor_values, dtype=torch.float32).unsqueeze(0)

    if glucose_tokens.size(1) < args.context_len:
        raise ValueError(f"输入序列长度不足 context-len={args.context_len}")

    generated_tokens = glucose_tokens[:, -args.context_len :]
    sensor_context = sensor_features[:, -args.context_len :, :]

    with torch.no_grad():
        for _ in range(args.predict_steps):
            token_input = generated_tokens[:, -config["max_seq_length"] :]
            sensor_input = sensor_context[:, -config["max_seq_length"] :, :]
            logits = model(token_input, sensor_input)[:, -1, :]
            next_token = sample_top_k(logits, args.top_k)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # 预测步无未来传感器观测，使用最后一个可用传感器向量进行延拓
            sensor_context = torch.cat([sensor_context, sensor_context[:, -1:, :]], dim=1)

    pred_tokens = generated_tokens[0, -args.predict_steps :].numpy()
    pred_glucose = pred_tokens + glucose_min

    out_df = pd.DataFrame(
        {
            "step": list(range(1, args.predict_steps + 1)),
            "predicted_token": pred_tokens,
            "predicted_glucose": pred_glucose,
        }
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"预测结果已保存: {args.output_csv}")


if __name__ == "__main__":
    main()
