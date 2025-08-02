import torch
from einops import rearrange

def generate_angles_2d(H,W,D, freq=None):
    freq = torch.tensor([10000**(-2*i/D) for i in range(int(D/2))]) if freq is None else freq
    pos = torch.outer(torch.linspace(-1, 1, steps=H),torch.linspace(-1, 1, steps=W))
    freq_tensor = torch.einsum("ij,k->ijk", pos, freq)
    return freq_tensor

def apply_angles_2d(x, f):
    x_reshaped = rearrange(x, "B h H W (D p) -> B h H W D p", p=2)
    real = x_reshaped[..., 0]
    imag = x_reshaped[..., 1]
    cosines, sines = f.cos(), f.sin()
    # r , i -> rcos-isin , rsin icos
    rot_real = real * cosines - imag * sines
    rot_imag = real * sines + imag * cosines
    rot_full = torch.concat((rot_real.unsqueeze(-1), rot_imag.unsqueeze(-1)), dim=-1)
    return rearrange(rot_full, "B h H W D p -> B h H W (D p)", p=2)

# Sanity Check :)
print(apply_angles_2d(torch.randn(1,8,64,64,768), generate_angles_2d(64,64,768)).shape)