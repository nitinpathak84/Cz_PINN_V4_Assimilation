import torch
import torch.nn as nn

class DriftNet(nn.Module):
    """
    t -> delta_eps_m(t), delta_eps_s(t) (bounded)
    """
    def __init__(self, hidden=64, layers=3, max_delta_eps=0.15):
        super().__init__()
        self.max_delta_eps = max_delta_eps

        net = []
        in_dim = 1
        for _ in range(max(layers - 1, 1)):
            net += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        net += [nn.Linear(in_dim, 2)]
        self.net = nn.Sequential(*net)

    def forward(self, t):
        out = torch.tanh(self.net(t)) * self.max_delta_eps
        return out[:, 0:1], out[:, 1:2]