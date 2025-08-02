import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from attention import ViTBlock

# Global Parameters
image_shape = 256
emb_dim = 768
patch_size = 16

class Encoder(nn.Module):
  def __init__(self, latent_dim, image_shape=image_shape, emb_dim=emb_dim, patch_size=patch_size, n_heads=8, dropout=0.1, layers=6, gaussian=False):
    super().__init__()
    self.patchifier = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
    self.Blocks = nn.ModuleList([ViTBlock(image_shape // patch_size, image_shape // patch_size, emb_dim, n_heads=8, dropout=dropout) for _ in range(layers)])
    self.ln = nn.LayerNorm(emb_dim)
    self.compress_latent = nn.Linear(emb_dim, latent_dim)

  def forward(self,x):
    x = self.patchifier(x)
    x = rearrange(x, "B D H W -> B (H W) D") # Flatten to B, N, D
    for vitBlock in self.Blocks:
      x = vitBlock(x)
    x = self.ln(x)
    x = self.compress_latent(x)
    return x

class Decoder(nn.Module):
  def __init__(self, latent_dim, image_shape=image_shape, emb_dim=emb_dim, patch_size=patch_size, n_heads=8, dropout=0.1, layers=6, gaussian=False):
    super().__init__()
    self.hw = image_shape // patch_size
    self.patch_size = patch_size
    self.decompress_latent = nn.Linear(latent_dim, emb_dim)
    self.ln = nn.LayerNorm(emb_dim)
    self.emb_to_patch = nn.Linear(emb_dim, 3*(patch_size**2))
    self.Blocks = nn.ModuleList([ViTBlock(image_shape // patch_size, image_shape // patch_size, emb_dim, n_heads=8, dropout=dropout) for _ in range(layers)])

  def forward(self,x):
    x = self.decompress_latent(x)
    for vitBlock in self.Blocks:
      x = vitBlock(x)
    self.ln(x)
    #shape is [B HW/p**2 (3 p p)]
    x = self.emb_to_patch(x)
    assert x.shape == torch.Size([x.shape[0], self.hw**2, 3*(self.patch_size**2)]), f"Expected shape {torch.Size([x.shape[0], self.hw**2, 3*(self.patch_size**2)])} got {x.shape}"
    x = rearrange(x, "B (H W) (D p1 p2) -> B D (H p1) (W p2)", H=self.hw, W=self.hw, p1=self.patch_size, p2=self.patch_size) # Expand to B, H, W, D
    return F.tanh(x)