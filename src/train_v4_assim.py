import os
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Adam, lr_scheduler

from physicsnemo.distributed import DistributedManager

from src.geometry import CzGeomParams, build_geometries
from src.sampling import make_volume_sampler, make_surface_sampler
from src.model_v4 import build_model
from src.drift_v4 import DriftNet
from src.sensors_v4 import SensorDataset
from src.losses_v4 import total_loss_v4_assim

def sample_time(cfg, n, device):
    t0, t1 = float(cfg.time.t_min), float(cfg.time.t_max)
    return (torch.rand(n, 1, device=device) * (t1 - t0) + t0)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config_v4_assim.yaml")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    os.makedirs(cfg.run.out_dir, exist_ok=True)

    # geometry
    p = CzGeomParams(**cfg.geometry)
    geoms = build_geometries(p)

    # model
    model = build_model(cfg.model.num_layers, cfg.model.layer_size, device=device)

    # drift net
    drift_net = DriftNet(
        hidden=cfg.drift.hidden,
        layers=cfg.drift.layers,
        max_delta_eps=cfg.drift.max_delta_eps
    ).to(device)

    # sensors + biases
    sensors = SensorDataset(
        meta_path=cfg.sensors.meta_path,
        ts_path=cfg.sensors.ts_path,
        id_col=cfg.sensors.id_col,
        time_col=cfg.sensors.time_col,
        value_col=cfg.sensors.value_col,
        device=device
    )
    bias_params = nn.Parameter(torch.zeros(sensors.num_sensors(), device=device))

    # optimizer
    params = list(model.parameters()) + [bias_params]
    if cfg.drift.enabled:
        params += list(drift_net.parameters())

    optimizer = Adam(params, lr=cfg.training.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cfg.training.lr_decay**step)

    # samplers
    melt_int = make_volume_sampler(geoms.melt, cfg.training.n_int_melt, device=device)
    crys_int = make_volume_sampler(geoms.crystal, cfg.training.n_int_crystal, device=device)

    melt_free_bc = make_surface_sampler(geoms.melt_free_band, cfg.training.n_bc_melt_free, device=device)
    crys_side_bc = make_surface_sampler(geoms.crystal_side_band, cfg.training.n_bc_crystal_side, device=device)
    iface_bc = make_surface_sampler(geoms.interface_band, cfg.training.n_bc_interface, device=device)
    axis_bc = make_surface_sampler(geoms.axis_band, cfg.training.n_bc_axis, device=device)

    # training log
    log_path = os.path.join(cfg.run.out_dir, "train_log_v4_assim.csv")
    with open(log_path, "w") as f:
        f.write("step,loss,pde_m,pde_s,rad_m,rad_s,intT,intF,axis,ic,sens,bias_reg,drift_reg,lr\n")

    for step in range(cfg.run.steps):
        mi_raw = next(iter(melt_int))[0]
        ci_raw = next(iter(crys_int))[0]
        mfb_raw = next(iter(melt_free_bc))[0]
        csb_raw = next(iter(crys_side_bc))[0]
        ifb_raw = next(iter(iface_bc))[0]
        ab_raw  = next(iter(axis_bc))[0]

        mi = {k: v.reshape(-1,1) for k,v in mi_raw.items()}
        ci = {k: v.reshape(-1,1) for k,v in ci_raw.items()}
        mfb = {k: v.reshape(-1,1) for k,v in mfb_raw.items()}
        csb = {k: v.reshape(-1,1) for k,v in csb_raw.items()}
        ifb = {k: v.reshape(-1,1) for k,v in ifb_raw.items()}
        ab  = {k: v.reshape(-1,1) for k,v in ab_raw.items()}

        # random time for collocation sets
        mi_t  = sample_time(cfg, mi["x"].shape[0], device)
        ci_t  = sample_time(cfg, ci["x"].shape[0], device)
        mfb_t = sample_time(cfg, mfb["x"].shape[0], device)
        csb_t = sample_time(cfg, csb["x"].shape[0], device)
        ifb_t = sample_time(cfg, ifb["x"].shape[0], device)
        ab_t  = sample_time(cfg, ab["x"].shape[0], device)

        # initial condition anchor uses t = t_min
        t0 = torch.full((mi["x"].shape[0],1), float(cfg.time.t_min), device=device)
        t0c = torch.full((ci["x"].shape[0],1), float(cfg.time.t_min), device=device)

        batches = {
            "mi_rzt": torch.cat([mi["x"], mi["y"], mi_t], dim=1),
            "mi_sdf": mi["sdf"],
            "ci_rzt": torch.cat([ci["x"], ci["y"], ci_t], dim=1),
            "ci_sdf": ci["sdf"],
            "mfb_rzt": torch.cat([mfb["x"], mfb["y"], mfb_t], dim=1),
            "csb_rzt": torch.cat([csb["x"], csb["y"], csb_t], dim=1),
            "ifb_rzt": torch.cat([ifb["x"], ifb["y"], ifb_t], dim=1),
            "ab_rzt":  torch.cat([ab["x"],  ab["y"],  ab_t],  dim=1),
            "mi_rz0":  torch.cat([mi["x"],  mi["y"],  t0],  dim=1),
            "ci_rz0":  torch.cat([ci["x"],  ci["y"],  t0c], dim=1),
        }

        sensor_batch = sensors.sample_batch(cfg.training.n_sensors_per_step, cfg.training.n_time_per_sensor)

        optimizer.zero_grad()
        loss, d = total_loss_v4_assim(model, drift_net, bias_params, batches, sensor_batch, cfg)
        loss.backward()

        if cfg.training.max_grad_norm and cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.training.max_grad_norm)

        optimizer.step()
        scheduler.step()

        if step % cfg.run.plot_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step} loss={d['loss_total'].item():.4e} "
                f"pde_m={d['pde_m'].item():.2e} pde_s={d['pde_s'].item():.2e} "
                f"rad_m={d['rad_m'].item():.2e} rad_s={d['rad_s'].item():.2e} "
                f"intT={d['intT'].item():.2e} intF={d['intF'].item():.2e} "
                f"axis={d['axis'].item():.2e} ic={d['ic'].item():.2e} "
                f"sens={d['sens'].item():.2e} lr={lr:.3e}"
            )
            with open(log_path, "a") as f:
                f.write(
                    f"{step},{d['loss_total'].item()},{d['pde_m'].item()},{d['pde_s'].item()},"
                    f"{d['rad_m'].item()},{d['rad_s'].item()},{d['intT'].item()},{d['intF'].item()},"
                    f"{d['axis'].item()},{d['ic'].item()},{d['sens'].item()},"
                    f"{d['bias_reg'].item()},{d['drift_reg'].item()},{lr}\n"
                )

            # checkpoint
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "drift": drift_net.state_dict() if cfg.drift.enabled else None,
                "bias": bias_params.detach().cpu(),
                "cfg": cfg,
            }
            torch.save(ckpt, os.path.join(cfg.run.out_dir, f"ckpt_v4_assim_{step}.pt"))

if __name__ == "__main__":
    main()
