# scripts/plot_v4_assim_dashboard.py
import os
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # safe for headless RunPod
import matplotlib.pyplot as plt

from src.model_v4 import build_model
from src.drift_v4 import DriftNet
from src.residuals_v4 import axisym_transient_component


def find_latest_ckpt(outputs_dir: str):
    ckpts = sorted(
        [f for f in os.listdir(outputs_dir) if f.startswith("ckpt_v4_assim_") and f.endswith(".pt")]
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {outputs_dir}")
    return os.path.join(outputs_dir, ckpts[-1])


def make_rz_grid(Rw, ztop, nr, nz):
    r = np.linspace(0.0, Rw, nr)
    z = np.linspace(0.0, ztop, nz)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    return r, z, rr, zz


def mask_melt(rr, zz, R_cr, h_m):
    return (rr <= R_cr) & (zz >= 0.0) & (zz <= h_m)


def mask_crystal(rr, zz, R_c, h_m, H_s):
    return (rr <= R_c) & (zz >= h_m) & (zz <= (h_m + H_s))


def overlay_core_geometry(ax, cfg):
    # Melt outline
    ax.plot(
        [0, cfg.geometry.R_cr, cfg.geometry.R_cr, 0, 0],
        [0, 0, cfg.geometry.h_m, cfg.geometry.h_m, 0],
        linewidth=1.5,
    )
    # Crystal outline
    ax.plot(
        [0, cfg.geometry.R_c, cfg.geometry.R_c, 0, 0],
        [
            cfg.geometry.h_m,
            cfg.geometry.h_m,
            cfg.geometry.h_m + cfg.geometry.H_s,
            cfg.geometry.h_m + cfg.geometry.H_s,
            cfg.geometry.h_m,
        ],
        linewidth=1.5,
    )


def eval_field_on_grid(model, device, rr, zz, tval):
    rr_t = torch.tensor(rr, dtype=torch.float32, device=device).reshape(-1, 1)
    zz_t = torch.tensor(zz, dtype=torch.float32, device=device).reshape(-1, 1)
    tt_t = torch.full_like(rr_t, float(tval))
    rzt = torch.cat([rr_t, zz_t, tt_t], dim=1)
    with torch.no_grad():
        out = model(rzt)
    Tm = out[:, 0:1].detach().cpu().numpy().reshape(rr.shape)
    Ts = out[:, 1:2].detach().cpu().numpy().reshape(rr.shape)
    return Tm, Ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=str, default="outputs", help="Outputs directory")
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint path")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    outputs_dir = args.outputs

    ckpt_path = args.ckpt if args.ckpt else find_latest_ckpt(outputs_dir)
    print("Using checkpoint:", ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    # Build and load model
    model = build_model(cfg.model.num_layers, cfg.model.layer_size, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Optional drift net
    drift = None
    if getattr(cfg.drift, "enabled", False) and ckpt.get("drift", None) is not None:
        drift = DriftNet(cfg.drift.hidden, cfg.drift.layers, cfg.drift.max_delta_eps).to(device)
        drift.load_state_dict(ckpt["drift"])
        drift.eval()

    # Grid
    nr, nz = int(cfg.inference.nr), int(cfg.inference.nz)
    r, z, rr, zz = make_rz_grid(cfg.geometry.R_w, cfg.geometry.z_top, nr, nz)

    # Times (t0, mid, end)
    t0 = float(cfg.time.t_min)
    t1 = float(cfg.time.t_max)
    times = [t0, 0.5 * (t0 + t1), t1]

    # Masks so the contour plots show only the physically-solved regions
    m_melt = mask_melt(rr, zz, cfg.geometry.R_cr, cfg.geometry.h_m)
    m_crys = mask_crystal(rr, zz, cfg.geometry.R_c, cfg.geometry.h_m, cfg.geometry.H_s)

    # Evaluate Tm/Ts on grid at three times
    Tm_list, Ts_list = [], []
    for tval in times:
        Tm, Ts = eval_field_on_grid(model, device, rr, zz, tval)
        Tm_plot = Tm.copy()
        Ts_plot = Ts.copy()
        Tm_plot[~m_melt] = np.nan
        Ts_plot[~m_crys] = np.nan
        Tm_list.append(Tm_plot)
        Ts_list.append(Ts_plot)

    # Diagnostics at t_end
    t_diag = t1

    # Interface temperature continuity along z=h_m, r in [0,R_c]
    r_if = np.linspace(0.0, cfg.geometry.R_c, 300)
    z_if = np.full_like(r_if, cfg.geometry.h_m)
    rzt_if = torch.tensor(
        np.stack([r_if, z_if, np.full_like(r_if, t_diag)], axis=1),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        out_if = model(rzt_if)
        Tm_if = out_if[:, 0].detach().cpu().numpy()
        Ts_if = out_if[:, 1].detach().cpu().numpy()

    # Melt free surface radiation check (z=h_m, r in [0,R_cr]) using Tm, Tz
    r_mf = np.linspace(0.0, cfg.geometry.R_cr, 300)
    z_mf = np.full_like(r_mf, cfg.geometry.h_m)
    rzt_mf = torch.tensor(
        np.stack([r_mf, z_mf, np.full_like(r_mf, t_diag)], axis=1),
        dtype=torch.float32,
        device=device,
    )
    T_mf, _, Tz_mf, _, _ = axisym_transient_component(model, rzt_mf, cfg.physics.r_eps, comp=0)
    q_cond_m = (-cfg.physics.k_m * Tz_mf).detach().cpu().numpy().reshape(-1)

    if drift is not None:
        with torch.no_grad():
            d_em, _ = drift(torch.full((1, 1), t_diag, device=device))
        eps_m = float(torch.clamp(cfg.physics.eps_m0 + d_em, 0.0, 1.0).item())
    else:
        eps_m = float(cfg.physics.eps_m0)

    q_rad_m = (
        eps_m
        * cfg.physics.sigma
        * (T_mf**4 - (float(cfg.physics.T_env) ** 4))
    ).detach().cpu().numpy().reshape(-1)

    # Crystal side radiation check (r=R_c, z in [h_m,h_m+H_s]) using Ts, Tr
    z_cs = np.linspace(cfg.geometry.h_m, cfg.geometry.h_m + cfg.geometry.H_s, 300)
    r_cs = np.full_like(z_cs, cfg.geometry.R_c)
    rzt_cs = torch.tensor(
        np.stack([r_cs, z_cs, np.full_like(z_cs, t_diag)], axis=1),
        dtype=torch.float32,
        device=device,
    )
    T_cs, Tr_cs, _, _, _ = axisym_transient_component(model, rzt_cs, cfg.physics.r_eps, comp=1)
    q_cond_s = (-cfg.physics.k_s * Tr_cs).detach().cpu().numpy().reshape(-1)

    if drift is not None:
        with torch.no_grad():
            _, d_es = drift(torch.full((1, 1), t_diag, device=device))
        eps_s = float(torch.clamp(cfg.physics.eps_s0 + d_es, 0.0, 1.0).item())
    else:
        eps_s = float(cfg.physics.eps_s0)

    q_rad_s = (
        eps_s
        * cfg.physics.sigma
        * (T_cs**4 - (float(cfg.physics.T_env) ** 4))
    ).detach().cpu().numpy().reshape(-1)

    # Build dashboard (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Row 1: Tm snapshots
    for j, tval in enumerate(times):
        ax = axes[0, j]
        im = ax.imshow(
            Tm_list[j],
            origin="lower",
            extent=[r.min(), r.max(), z.min(), z.max()],
            aspect="auto",
        )
        overlay_core_geometry(ax, cfg)
        ax.set_title(f"Tm(r,z) @ t={tval:.0f}s")
        ax.set_xlabel("r (m)")
        ax.set_ylabel("z (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Ts snapshots
    for j, tval in enumerate(times):
        ax = axes[1, j]
        im = ax.imshow(
            Ts_list[j],
            origin="lower",
            extent=[r.min(), r.max(), z.min(), z.max()],
            aspect="auto",
        )
        overlay_core_geometry(ax, cfg)
        ax.set_title(f"Ts(r,z) @ t={tval:.0f}s")
        ax.set_xlabel("r (m)")
        ax.set_ylabel("z (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: diagnostics
    ax = axes[2, 0]
    ax.plot(r_if, Tm_if, label="Tm on interface")
    ax.plot(r_if, Ts_if, label="Ts on interface")
    ax.set_title("Interface temperature continuity (z=h_m)")
    ax.set_xlabel("r (m)")
    ax.set_ylabel("T (K)")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(r_mf, q_cond_m, label="q_cond = -k_m*Tz")
    ax.plot(r_mf, q_rad_m, label="q_rad = eps*sigma*(T^4 - Tenv^4)")
    ax.set_title(f"Melt free surface radiation check @ t={t_diag:.0f}s (eps_m={eps_m:.3f})")
    ax.set_xlabel("r (m)")
    ax.set_ylabel("Heat flux (W/m^2)")
    ax.legend()

    ax = axes[2, 2]
    ax.plot(z_cs, q_cond_s, label="q_cond = -k_s*Tr")
    ax.plot(z_cs, q_rad_s, label="q_rad = eps*sigma*(T^4 - Tenv^4)")
    ax.set_title(f"Crystal side radiation check @ t={t_diag:.0f}s (eps_s={eps_s:.3f})")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("Heat flux (W/m^2)")
    ax.legend()

    plt.tight_layout()

    step = ckpt.get("step", "na")
    out_path = os.path.join(outputs_dir, f"v4_assim_thermal_dashboard_step_{step}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print("Saved dashboard:", out_path)


if __name__ == "__main__":
    main()
