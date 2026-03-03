# src/plotting.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def make_grid(R_w: float, z_top: float, nr: int, nz: int, device):
    r = np.linspace(0.0, R_w, nr)
    z = np.linspace(0.0, z_top, nz)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    rr_t = torch.from_numpy(rr).float().to(device).reshape(-1, 1)
    zz_t = torch.from_numpy(zz).float().to(device).reshape(-1, 1)
    coords = torch.cat([rr_t, zz_t], dim=1)
    return coords, (r, z)


def save_T_plot(T_grid, r, z, title: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(
        T_grid,
        origin="lower",
        extent=[r.min(), r.max(), z.min(), z.max()],
        aspect="auto",
    )
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)