import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, queries, keys, values):
        attn_out, _ = self.attn(queries, keys, values)
        x = self.ln1(queries + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class SimplePerceiverIO(nn.Module):
    """
    Perceiver-style multimodal encoder:
    - learnable latent array
    - cross-attention from latents to inputs
    - latent transformer blocks
    """

    def __init__(self, input_dim: int, latent_dim: int = 256, num_latents: int = 128, num_layers: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.cross = CrossAttentionBlock(latent_dim, num_heads=8)
        self.latent_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=8,
                    dim_feedforward=latent_dim * 4,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.output_head = nn.Linear(latent_dim, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, input_dim) concatenated multimodal tokens
        """
        B, N, _ = x.shape
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        inp = self.input_proj(x)
        latents = self.cross(latents, inp, inp)
        for layer in self.latent_layers:
            latents = layer(latents)
        pooled = latents.mean(dim=1)
        return self.output_head(pooled)
