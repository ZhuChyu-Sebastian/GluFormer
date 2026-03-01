import math
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """Autoregressive transformer used by GluFormer (single-modality glucose tokens)."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_heads: int,
        n_layers: int,
        max_seq_length: int,
        dropout: float,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.register_buffer(
            "pos_embedding", self._create_pos_embedding(max_seq_length, n_embd), persistent=False
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.linear = nn.Linear(n_embd, vocab_size)

    @staticmethod
    def _create_pos_embedding(max_seq_length: int, n_embd: int) -> torch.Tensor:
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd))
        pos_embedding = torch.zeros(max_seq_length, n_embd)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        token_embeddings = self.embedding(tokens)
        position_embeddings = self.pos_embedding[: tokens.size(1), :].unsqueeze(0)
        embeddings = token_embeddings + position_embeddings

        seq_length = tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=tokens.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))

        hidden = self.transformer(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )
        return self.linear(hidden)


class CrossModalGluFormer(nn.Module):
    """Dual-tower GluFormer with cross-modal attention (glucose tower attends to sensor tower)."""

    def __init__(
        self,
        vocab_size: int,
        sensor_dim: int,
        n_embd: int,
        n_heads: int,
        n_layers: int,
        max_seq_length: int,
        dropout: float,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.glucose_embedding = nn.Embedding(vocab_size, n_embd)
        self.sensor_projection = nn.Sequential(
            nn.Linear(sensor_dim, n_embd),
            nn.LayerNorm(n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "pos_embedding", TransformerModel._create_pos_embedding(max_seq_length, n_embd), persistent=False
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fuse_norm = nn.LayerNorm(n_embd)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, glucose_tokens: torch.Tensor, sensor_features: torch.Tensor) -> torch.Tensor:
        glucose_x = self.glucose_embedding(glucose_tokens)
        sensor_x = self.sensor_projection(sensor_features)

        position = self.pos_embedding[: glucose_tokens.size(1), :].unsqueeze(0)
        glucose_x = glucose_x + position
        sensor_x = sensor_x + position

        attended, _ = self.cross_attention(query=glucose_x, key=sensor_x, value=sensor_x, need_weights=False)
        fused = self.fuse_norm(glucose_x + attended)

        seq_length = glucose_tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=glucose_tokens.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))

        hidden = self.transformer(fused, mask=causal_mask)
        return self.lm_head(hidden)
