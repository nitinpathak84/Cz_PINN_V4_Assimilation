# src/visualize_geometry.py
import os
import hydra
import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe

from src.geometry import CzGeomParams, build_geometries


def _sample_points(datapipe: GeometryDatapipe, keys=("x", "y"), nmax=4000):
    """Grab one batch from a GeometryDatapipe and return x,y as numpy arrays."""
    batch = next(iter(datapipe))[0]
    x = batch[keys[0]].reshape(-1).detach().cpu().numpy()
    y = batch[keys[1]].reshape(-1).detach().cpu().numpy()
    if len(x) > nmax:
        idx = np.random.choice(len(x), size=nmax, replace=False)
        x, y = x[idx], y[idx]
    return x, y


@hydra.main(version_base="1.3", config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    os.makedirs(cfg.run.out_dir, exist_ok=True)

    # Build geometries (same ones used in training)
    p = CzGeomParams(**cfg.geometry)
    geoms = build_geometries(p)

    # Create datapipes (same sampling types used in training)
    melt_int = GeometryDatapipe(
        geom_objects=[geoms.melt],
        batch_size=1,
        num_points=cfg.training.n_int_melt,
        sample_type="volume",
        device=device,
        num_workers=1,
        requested_vars=["x", "y", "sdf"],
    )
    crys_int = GeometryDatapipe(
        geom_objects=[geoms.crystal],
        batch_size=1,
        num_points=cfg.training.n_int_crystal,
        sample_type="volume",
        device=device,
        num_workers=1,
        requested_vars=["x", "y", "sdf"],
    )
    heat_bc = GeometryDatapipe(
        geom_objects=[geoms.heater_band],
        batch_size=1,
        num_points=cfg.training.n_bc_heat,
        sample_type="surface",
        device=device,
        num_workers=1,
        requested_vars=["x", "y"],
    )
    cool_bc = GeometryDatapipe(
        geom_objects=[geoms.cool_band],
        batch_size=1,
        num_points=cfg.training.n_bc_cool,
        sample_type="surface",
        device=device,
        num_workers=1,
        requested_vars=["x", "y"],
    )
    axis_bc = GeometryDatapipe(
        geom_objects=[geoms.axis_band],
        batch_size=1,
        num_points=cfg.training.n_bc_axis,
        sample_type="surface",
        device=device,
        num_workers=1,
        requested_vars=["x", "y"],
    )

    # Sample points
    xm, zm = _sample_points(melt_int, nmax=5000)
    xc, zc = _sample_points(crys_int, nmax=5000)
    xh, zh = _sample_points(heat_bc, nmax=3000)
    xw, zw = _sample_points(cool_bc, nmax=3000)
    xa, za = _sample_points(axis_bc, nmax=3000)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    # --- Outline the key rectangles (domain geometry) ---
    # Melt rectangle
    ax.plot([0, p.R_cr, p.R_cr, 0, 0], [0, 0, p.h_m, p.h_m, 0], linewidth=2, label="Melt domain")
    # Crystal rectangle
    ax.plot([0, p.R_c, p.R_c, 0, 0], [p.h_m, p.h_m, p.h_m + p.H_s, p.h_m + p.H_s, p.h_m],
            linewidth=2, label="Crystal domain")

    # Heater band rectangle
    ax.plot([p.R_h - p.bc_thickness, p.R_h + p.bc_thickness, p.R_h + p.bc_thickness, p.R_h - p.bc_thickness, p.R_h - p.bc_thickness],
            [p.z_h1, p.z_h1, p.z_h2, p.z_h2, p.z_h1], linewidth=1.5, label="Heater band")

    # Cooling band rectangle
    ax.plot([p.R_w - p.bc_thickness, p.R_w + p.bc_thickness, p.R_w + p.bc_thickness, p.R_w - p.bc_thickness, p.R_w - p.bc_thickness],
            [p.z_w1, p.z_w1, p.z_w2, p.z_w2, p.z_w1], linewidth=1.5, label="Cooling band")

    # Axis band rectangle
    ax.plot([0, p.bc_thickness, p.bc_thickness, 0, 0],
            [0, 0, p.z_top, p.z_top, 0], linewidth=1.5, label="Axis band")

    # --- Point clouds ---
    ax.scatter(xm, zm, s=2, alpha=0.5, label="Interior: melt")
    ax.scatter(xc, zc, s=2, alpha=0.5, label="Interior: crystal")
    ax.scatter(xh, zh, s=6, alpha=0.9, label="BC: heater")
    ax.scatter(xw, zw, s=6, alpha=0.9, label="BC: cooling")
    ax.scatter(xa, za, s=6, alpha=0.9, label="BC: axis")

    ax.set_xlim(0, p.R_w)
    ax.set_ylim(0, p.z_top)
    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Cz PINN geometry + sampled point clouds")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    out_path = os.path.join(cfg.run.out_dir, "geometry_pointcloud.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()