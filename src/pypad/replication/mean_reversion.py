import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal
from torch.optim import Adam, SGD

device = "cuda" if torch.cuda.is_available() else "cpu"


def ornstein_uhlenbeck(r0, theta, mu, sigma, t, n_path):
    n_steps = len(t)
    zeros = torch.tensor([0], device=device)
    time_increments = torch.diff(t, prepend=zeros)
    normals = torch.randn((n_path, n_steps), device=device)

    r = torch.zeros_like(normals) + r0
    for i, dt in enumerate(time_increments):
        if i == 0:
            continue
        dr = theta * (mu - r[:, i - 1]) * dt + sigma * torch.sqrt(dt) * normals[:, i]
        r[:, i] = r[:, i - 1] + dr

    return r


def entropic_risk_measure(profit, a: float = 1.0):
    n = len(profit)
    return (torch.logsumexp(-a * profit, dim=0) - math.log(n)) / a


class MLPHedgingStrategy(nn.Module):
    def __init__(
        self,
        num_inputs: int = 1,
        num_hidden_layers: int = 4,
        num_features: int = 32,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()

        self.input_layer = nn.Linear(num_inputs, num_features)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(num_features, num_features))
            self.hidden_layers.append(nn.ReLU(inplace=True))

        self.output_layer = nn.Linear(num_features, num_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

    def get_hedge_ratio(self, z_spreads: torch.Tensor):
        input = z_spreads.unsqueeze(-1)
        return self(input).squeeze(-1)


class BandedMLPHedgingStrategy(nn.Module):
    def __init__(
        self,
        num_inputs: int = 2,
        num_hidden_layers: int = 4,
        num_features: int = 32,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()

        self.input_layer = nn.Linear(num_inputs, num_features)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(num_features, num_features))
            self.hidden_layers.append(nn.ReLU(inplace=True))

        self.output_layer = nn.Linear(num_features, num_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

    def get_hedge_ratio(self, z_spreads: torch.Tensor):
        outputs = []
        unsq_z_spreads = z_spreads.unsqueeze(-1)
        for i in range(z_spreads.shape[1]):
            feature1 = unsq_z_spreads[:, i, ...]
            if i == 0:
                feature2 = torch.zeros_like(feature1)
            else:
                feature2 = outputs[-1] * 100

            input = torch.cat([feature1, feature2], dim=-1)
            output = self(input)
            outputs.append(output)

        return torch.cat(outputs, dim=-1)


def run() -> None:
    ONE_BP = 1e-4
    r0 = 30 * ONE_BP
    mu = 30 * ONE_BP
    theta = 1
    sigma = 60 * ONE_BP
    T = 10
    num_steps = 250 * T
    r = 0.04
    n_path = 1_000
    risk_aversion = 1

    mlp = MLPHedgingStrategy(num_hidden_layers=4).to(device)

    t = torch.linspace(0, T, num_steps + 1, device=device)
    z_spreads = ornstein_uhlenbeck(r0, theta, mu, sigma, t, n_path)
    S = z_spreads * 10_000

    change_in_prices = torch.diff(S, axis=1)
    payoff = S[:, -1] - S[:, 0]

    delta = 0.5 + 0.25 * torch.randn((n_path, num_steps), device=device)  # final timestep dropped
    profits = change_in_prices * delta

    pnl = profits.sum(axis=1) - payoff
    risk_measure = entropic_risk_measure(pnl, a=risk_aversion)
    print(f"RAND: {risk_measure}")

    params = mlp.parameters()
    optimizer = Adam(params=params, lr=0.0001, weight_decay=0.000001)
    mlp.train()
    for epoch in range(1_000):
        z_spreads = ornstein_uhlenbeck(r0, theta, mu, sigma, t, n_path)
        S = z_spreads * 10_000
        change_in_prices = S.diff(axis=1)
        delta = mlp.get_hedge_ratio(S)[:, :-1]
        profits = change_in_prices * delta
        t_cost = (S[:, 1:-1] * delta.diff(axis=1)).abs().sum(dim=1) * 0.01
        pnl = profits.sum(axis=1) - payoff - t_cost
        risk_measure = entropic_risk_measure(pnl, a=risk_aversion)

        optimizer.zero_grad()
        risk_measure.backward()
        optimizer.step()
        print(f"{epoch}: Measure={risk_measure:.2f}, Profit={pnl.mean():.2f}, Cost={t_cost.mean():.2f}")

    mlp.eval()
    prices = torch.arange(601, dtype=torch.float32, device=device) - 300.0
    prices = prices.unsqueeze(-1)
    delta = mlp.get_hedge_ratio(prices)
    np.savetxt("output.csv", delta.detach().cpu().numpy(), delimiter=",")

    banded_mlp = BandedMLPHedgingStrategy(num_hidden_layers=4).to(device)
    delta = banded_mlp.get_hedge_ratio(S)[:, :-1]

    t_costs = 0.01

    banded_params = banded_mlp.parameters()
    banded_optimizer = Adam(params=banded_params, lr=0.0001, weight_decay=0.000001)
    banded_mlp.train()
    for epoch in range(1_000):
        z_spreads = ornstein_uhlenbeck(r0, theta, mu, sigma, t, n_path)
        S = z_spreads * 10_000
        change_in_prices = S.diff(axis=1)
        delta = banded_mlp.get_hedge_ratio(S)[:, :-1]
        profits = change_in_prices * delta
        t_cost = (S[:, 1:-1] * delta.diff(axis=1)).abs().sum(dim=1) * t_costs
        pnl = profits.sum(axis=1) - payoff - t_cost
        risk_measure = entropic_risk_measure(pnl, a=risk_aversion)

        banded_optimizer.zero_grad()
        risk_measure.backward()
        banded_optimizer.step()
        print(f"{epoch}: Measure={risk_measure:.2f}, Profit={pnl.mean():.2f}, Cost={t_cost.mean():.2f}")

    # This probably doesn't work...
    banded_mlp.eval()
    prices = torch.arange(601, dtype=torch.float32, device=device) - 300.0
    prices = prices.unsqueeze(-1).expand(-1, 100)
    delta = banded_mlp.get_hedge_ratio(prices)
    np.savetxt("output_banded.csv", delta[:, -1].detach().cpu().numpy(), delimiter=",")


if __name__ == "__main__":
    run()
