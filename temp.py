from dataclasses import dataclass
from typing import Protocol, Sequence

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal


def gbm(S0, r, sigma, t, n_path):
    drift = r * t
    time_steps = torch.diff(t, prepend=torch.tensor([0]))
    normals = torch.randn((n_path, num_steps + 1))
    normals = torch.cat([normals, -normals], dim=0)
    vol = sigma * torch.sqrt(time_steps) * normals.cumsum(axis=1)
    exponent = drift + vol
    paths = S0 * torch.exp(exponent)
    return paths


def _d1(S, K, r, sigma, t, T):
    t2m = T - t
    numerator = (S / K).log() + (r + 0.5 * sigma**2) * t2m
    denominator = sigma * torch.sqrt(t2m)
    return numerator / denominator


def _d2(d1, sigma, t, T):
    t2m = T - t
    return d1 - sigma * torch.sqrt(t2m)


def call_price(S, K, r, t, T, d1, d2):
    normal = Normal(0, 1)
    return S * normal.cdf(d1) - K * torch.exp(-r * (T - t)) * normal.cdf(d2)


def put_price(S, K, r, t, T, d1, d2):
    normal = Normal(0, 1)
    return -S * normal.cdf(-d1) + K * torch.exp(-r * (T - t)) * normal.cdf(-d2)


def call_delta(d1):
    normal = Normal(0, 1)
    return normal.cdf(d1)


def put_delta(d1):
    normal = Normal(0, 1)
    return normal.cdf(d1) - 1


class Stock:
    def __init__(
        self,
        S0: float,
        mu: float,
        r: float,
        sigma: float,
        time_steps: Sequence[float],
        cost: float = 0.0,
    ) -> None:
        self.S0 = S0
        self.mu = mu
        self.r = r
        self.sigma = sigma
        self.time_steps = time_steps
        self.cost = cost

        self.prices: np.ndarray | None = None

    def simulate(self, num_sims: int) -> None:
        drift = self.mu * self.time_steps
        time_increments = torch.diff(t, prepend=torch.tensor([0]))
        normals = torch.randn((num_sims // 2, num_steps + 1))
        normals = torch.cat([normals, -normals], dim=0)
        vol = self.sigma * torch.sqrt(time_increments) * normals.cumsum(axis=1)
        exponent = drift + vol
        self.prices = self.S0 * torch.exp(exponent)

    def discount_factors(self) -> None:
        discount_factors = torch.exp(-self.r * self.time_steps)
        return discount_factors

    def prices_at(self, t: float):
        return self.prices[:, self.time_steps == t].squeeze()


@dataclass
class EuropeanOption:
    stock: Stock
    strike: float
    maturity: float
    is_call: bool

    def payout(self):
        S = self.stock.prices_at(self.maturity)
        if self.is_call:
            return torch.relu(S - self.strike)
        return torch.relu(self.strike - S)


class HedgingStrategy(Protocol):
    @property
    def stock(self) -> Stock:
        ...

    @property
    def derivative(self) -> EuropeanOption:
        ...

    def get_hedge_ratio(self, time_step_idx: int | None = None):
        ...


@dataclass
class BlackScholesHedgingStrategy:
    option: EuropeanOption

    @property
    def stock(self) -> Stock:
        return self.option.stock

    @property
    def derivative(self) -> EuropeanOption:
        return self.option

    def get_hedge_ratio(self, time_step_idx: int | None = None):
        if time_step_idx:
            raise NotImplementedError("TODO")

        stock = self.option.stock
        S = stock.prices
        K = self.option.strike
        r = stock.r
        sigma = stock.sigma
        d = BlackScholesHedgingStrategy.d1(S, K, r, sigma, stock.time_steps, self.option.maturity)

        if self.option.is_call:
            return BlackScholesHedgingStrategy.call_delta(d)

        return BlackScholesHedgingStrategy.put_delta(d)

    @staticmethod
    def d1(S, K, r, sigma, t, T):
        t2m = T - t
        numerator = (S / K).log() + (r + 0.5 * sigma**2) * t2m
        denominator = sigma * torch.sqrt(t2m)
        return numerator / denominator

    @staticmethod
    def d2(d1, sigma, t, T):
        t2m = T - t
        return d1 - sigma * torch.sqrt(t2m)

    @staticmethod
    def call_price(S, K, r, t, T, d1, d2):
        normal = Normal(0, 1)
        return S * normal.cdf(d1) - K * torch.exp(-r * (T - t)) * normal.cdf(d2)

    @staticmethod
    def put_price(S, K, r, t, T, d1, d2):
        normal = Normal(0, 1)
        return -S * normal.cdf(-d1) + K * torch.exp(-r * (T - t)) * normal.cdf(-d2)

    @staticmethod
    def call_delta(d1):
        normal = Normal(0, 1)
        return normal.cdf(d1)

    @staticmethod
    def put_delta(d1):
        normal = Normal(0, 1)
        return normal.cdf(d1) - 1


class Pricer:
    def price(self, hedging_strategy: HedgingStrategy) -> float:
        S = hedging_strategy.stock.prices
        discount_factors = hedging_strategy.stock.discount_factors()
        change_in_prices = torch.diff(discount_factors * S, axis=1)

        delta = hedging_strategy.get_hedge_ratio()[:, :-1]
        discounted_profits = change_in_prices * delta

        payoff = hedging_strategy.option.payout()
        cost_of_hedging = math.exp(-r * T) * payoff - discounted_profits.sum(axis=1)
        return cost_of_hedging.mean()


    def fit(self, hedging_strategy: HedgingStrategy) -> float:

        params = hedging_strategy.parameters()
        optimizer = Adam(params=params)

        hedging_strategy.train()
        for epoch in range(100):
            S = hedging_strategy.stock.prices
            discount_factors = hedging_strategy.stock.discount_factors()
            change_in_prices = torch.diff(discount_factors * S, axis=1)

            delta = hedging_strategy.get_hedge_ratio()[:, :-1]
            discounted_profits = change_in_prices * delta

            payoff = hedging_strategy.option.payout()
            cost_of_hedging = math.exp(-r * T) * payoff - discounted_profits.sum(axis=1)
            pnl = -cost_of_hedging
            risk_measure = self.entropic_risk_measure(pnl)
            optimizer.zero_grad()
            risk_measure.backward()
            optimizer.step()
            print(f'{epoch}: {risk_measure}')

        return cost_of_hedging.mean()

    def entropic_risk_measure(self, x, a:float=1):
        n = len(x)
        return (torch.logsumexp(-a * x, dim=0) - math.log(n)) / a

class MLPHedgingStrategy(nn.Module):
    def __init__(
        self,
        option: EuropeanOption,
        num_inputs: int = 4,
        num_hidden_layers: int = 4,
        num_features: int = 32,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()

        self.option = option

        # Define the input layer
        self.input_layer = nn.Linear(num_inputs, num_features)

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(num_features, num_features))
            self.hidden_layers.append(nn.ReLU(inplace=True))

        # Define the output layer
        self.output_layer = nn.Linear(num_features, num_outputs)

    @property
    def stock(self) -> Stock:
        return self.option.stock

    @property
    def derivative(self) -> EuropeanOption:
        return self.option

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_hedge_ratio(self, time_step_idx: int | None = None):
        S = self.stock.prices
        K = self.option.strike
        T = self.option.maturity
        t = self.stock.time_steps
        log_moneyness = torch.log(S / K).unsqueeze(-1)
        time_to_mat = (T - t).expand(S.shape[0], -1).unsqueeze(-1)
        rfr = (torch.zeros_like(S) + stock.r).unsqueeze(-1)
        vol = (torch.zeros_like(S) + stock.sigma).unsqueeze(-1)
        input = torch.cat([log_moneyness, time_to_mat, rfr, vol], dim=-1)
        return self(input).squeeze(-1)


if __name__ == "__main__":
    S0 = 100
    K = 108
    T = 1
    num_steps = 100
    sigma = 0.2
    mu = 0.08
    r = 0.04
    n_path = 10_000

    t = torch.linspace(0, T, num_steps + 1)

    if True:
        S = gbm(S0, r, sigma, t, n_path)
        d1 = _d1(S, K, r, sigma, t, T)
        d2 = _d2(d1, sigma, t, T)
        c = call_price(S, K, r, t, T, d1, d2)
        p = put_price(S, K, r, t, T, d1, d2)

        print(f"Theoretical Call Price={c[0, 0]}")
        print(f"Theoretical Put  Price={p[0, 0]}\n")

        delta = call_delta(d1)[:, :-1]

        # Method 1 (Most intuitive to me...)
        # Cost of replication is the sum of each transaction plus the final payout
        discount_factors = torch.exp(-r * t)
        zeros = torch.zeros_like(delta[:, :1])
        txns = S * delta.diff(axis=1, prepend=zeros, append=zeros)
        call_payoff = torch.relu(S[:, -1] - K)
        txns[:, -1] += call_payoff
        disounted_txns = discount_factors * txns
        cost_of_hedging = disounted_txns.sum(axis=1)
        print(f"Black-Scholes call price via replication = {cost_of_hedging.mean()}")

        txns = S * (delta - 1).diff(axis=1, prepend=zeros, append=zeros)
        put_payoff = torch.relu(K - S[:, -1])
        txns[:, -1] += put_payoff
        disounted_txns = discount_factors * txns
        cost_of_hedging = disounted_txns.sum(axis=1)
        print(f"Black-Scholes put  price via replication = {cost_of_hedging.mean()}")

        # Method 2
        # Cost of replication is the final payout less any gains from stock growth
        discounted_profits = torch.diff(discount_factors * S, axis=1) * delta
        cost_of_hedging = math.exp(-r * T) * call_payoff - discounted_profits.sum(axis=1)
        print(f"Black-Scholes call price via replication = {cost_of_hedging.mean()}")

        discounted_profits = torch.diff(discount_factors * S, axis=1) * (delta - 1)
        cost_of_hedging = math.exp(-r * T) * put_payoff - discounted_profits.sum(axis=1)
        print(f"Black-Scholes put  price via replication = {cost_of_hedging.mean()}")

        """
        Method equivalence:
        ∫ d(ΔS) = ∫ Δ dS/dt dt + ∫ S dΔ/dt dt
        """

    stock = Stock(S0=S0, mu=r, r=r, sigma=sigma, time_steps=t)
    stock.simulate(n_path)
    stock.prices

    call_option = EuropeanOption(stock, K, T, is_call=True)
    bs_strategy = BlackScholesHedgingStrategy(call_option)

    pricer = Pricer()
    bs_call_price = pricer.price(bs_strategy)
    print(f"Black-Scholes call price via replication = {bs_call_price}")

    # Create an instance of the MLPHedgingStrategy
    num_inputs = 4
    num_hidden_layers = 4
    num_features = 32
    num_outputs = 1
    mlp_strategy = MLPHedgingStrategy(
        option=call_option,
        num_inputs=num_inputs,
        num_hidden_layers=num_hidden_layers,
        num_features=num_features,
        num_outputs=num_outputs,
    )

    # Check setup is as expected
    example_input = torch.rand(n_path, len(t), num_inputs)  # Random example input data
    output = mlp_strategy(example_input).squeeze(-1)
    print(f"Simple test returns output of shape {output.shape}")

    pricer = Pricer()
    mlp_call_price = pricer.price(mlp_strategy)
    print(f"MLP call price via replication = {mlp_call_price}")


    pricer.fit(mlp_strategy)