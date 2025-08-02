from RoPE import apply_angles_2d, generate_angles_2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, H,W, emb_dim, n_heads=8):
        super().__init__()
        self.H = H
        self.W = W
        self.n_heads = n_heads
        head_dim = emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3*emb_dim, bias=False)
        self.apply_angles_2d = apply_angles_2d
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.register_buffer("freq", generate_angles_2d(H, W, head_dim), persistent=False)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # to 2D
        q = rearrange(q, "B (H W) (h D) -> B h H W D", H=self.H, W=self.W, h=self.n_heads)
        k = rearrange(k, "B (H W) (h D) -> B h H W D", H=self.H, W=self.W, h=self.n_heads)
        v = rearrange(v, "B (H W) (h D) -> B h H W D", H=self.H, W=self.W, h=self.n_heads)

        q = apply_angles_2d(q, self.freq)
        k = apply_angles_2d(k, self.freq)
        v = apply_angles_2d(v, self.freq)

        # to 1D
        q = rearrange(q, "B h H W D -> B h (H W) D", H=self.H, W=self.W, h=self.n_heads)
        k = rearrange(k, "B h H W D -> B h (H W) D", H=self.H, W=self.W, h=self.n_heads)
        v = rearrange(v, "B h H W D -> B h (H W) D", H=self.H, W=self.W, h=self.n_heads)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B h N D -> B N (h D)")
        x = self.proj(x)
        return x

class ViTBlock(nn.Module):
  def __init__(self, H, W, emb_dim, n_heads=8, dropout=0.1):
    self.H, self.W, self.emb_dim = H, W, emb_dim
    super().__init__()
    self.attn = nn.Sequential(nn.LayerNorm(emb_dim),
                              Attention(H,W,emb_dim,n_heads=n_heads))
    self.MLP = nn.Sequential(nn.LayerNorm(emb_dim),
                             nn.Linear(emb_dim, emb_dim*4, bias=True),
                             nn.GELU(),
                             nn.Dropout(dropout),
                             nn.Linear(emb_dim*4, emb_dim, bias=True),
                             nn.Dropout(dropout))
  def forward(self, x):
    assert x.ndim == 3, f"Expected shape [B, N, D], but got shape {x.shape}. You probably passed [B, H, W, D] instead."
    assert x.shape == torch.Size([x.shape[0], self.H * self.W, self.emb_dim]), f"Expected shape [B, N, D] -> {torch.Size([x.shape[0], self.H * self.W, self.emb_dim])}, got {x.shape}"
    x = x + self.attn(x)
    x = x + self.MLP(x)
    return x
  
# Sanity Check :)
print(ViTBlock(64,64,384)(torch.randn(1, 64**2, 384)).shape)