import torch

def axisym_transient_component(model, rzt, r_eps: float, comp: int):
    """
    rzt: [N,3] columns [r,z,t]
    comp: 0->Tm, 1->Ts
    returns: T, Tr, Tz, Tt, lap_axisym
    """
    r = rzt[:, 0:1].detach().clone().requires_grad_(True)
    z = rzt[:, 1:2].detach().clone().requires_grad_(True)
    t = rzt[:, 2:3].detach().clone().requires_grad_(True)

    out = model(torch.cat([r, z, t], dim=1))
    T = out[:, comp:comp+1]

    Tr = torch.autograd.grad(T, r, torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    Tz = torch.autograd.grad(T, z, torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    Tt = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True, retain_graph=True)[0]

    Trr = torch.autograd.grad(Tr, r, torch.ones_like(Tr), create_graph=True, retain_graph=True)[0]
    Tzz = torch.autograd.grad(Tz, z, torch.ones_like(Tz), create_graph=True, retain_graph=True)[0]

    r_safe = torch.clamp(r, min=r_eps)
    lap = Trr + (1.0 / r_safe) * Tr + Tzz

    return T, Tr, Tz, Tt, lap