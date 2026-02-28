import argparse
from pathlib import Path

import pandas as pd
import torch

from train_model.modeling import TransformerModel


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
    parser = argparse.ArgumentParser(description="使用训练好的 GluFormer 预测未来 CGM token")
    parser.add_argument("--checkpoint", type=Path, default=Path("train_model/physiocgm_gluformer.pt"))
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
    model = TransformerModel(
        vocab_size=config["vocab_size"],
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
    tokens = torch.tensor((values - glucose_min).to_numpy(), dtype=torch.long).unsqueeze(0)

    if tokens.size(1) < args.context_len:
        raise ValueError(f"输入序列长度不足 context-len={args.context_len}")

    generated = tokens[:, -args.context_len :]
    with torch.no_grad():
        for _ in range(args.predict_steps):
            logits = model(generated[:, -config["max_seq_length"] :])[:, -1, :]
            next_token = sample_top_k(logits, args.top_k)
            generated = torch.cat([generated, next_token], dim=1)

    pred_tokens = generated[0, -args.predict_steps :].numpy()
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
