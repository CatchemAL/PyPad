from dataclasses import dataclass
from math import sqrt
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.stats import norm


def gbm(S0, r, sigma, t, n_path):
    drift = r * t
    time_steps = np.diff(t, prepend=0)
    normals = np.random.standard_normal((n_path, num_steps + 1))
    normals = np.vstack([normals, -normals])
    vol = sigma * np.sqrt(time_steps) * normals.cumsum(axis=1)
    exponent = drift + vol
    paths = S0 * np.exp(exponent)
    return paths


def _d1(S, K, r, sigma, t, T):
    t2m = T - t
    numerator = np.log(S / K) + (r + 0.5 * sigma**2) * t2m
    denominator = sigma * np.sqrt(t2m)
    with np.errstate(divide="ignore"):
        return numerator / denominator


def _d2(d1, sigma, t, T):
    t2m = T - t
    return d1 - sigma * np.sqrt(t2m)


def call_price(S, K, r, t, T, d1, d2):
    return S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)


def put_price(S, K, r, t, T, d1, d2):
    return -S * norm.cdf(-d1) + K * np.exp(-r * (T - t)) * norm.cdf(-d2)


def call_delta(d1):
    return norm.cdf(d1)


def put_delta(d1):
    return norm.cdf(d1) - 1


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
        time_increments = np.diff(self.time_steps, prepend=0)
        normals = np.random.standard_normal((num_sims // 2, len(self.time_steps)))
        normals = np.vstack([normals, -normals])
        vol = self.sigma * np.sqrt(time_increments) * normals.cumsum(axis=1)
        exponent = drift + vol
        self.prices = self.S0 * np.exp(exponent)

    def discount_factors(self) -> None:
        discount_factors = np.exp(-self.r * self.time_steps)
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
            return np.maximum(S - self.strike, 0)
        return np.maximum(self.strike - S, 0)


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
        r = stock.r
        K = self.option.strike
        d = BlackScholesHedgingStrategy.d1(
            S, K, stock.r, stock.sigma, stock.time_steps, self.option.maturity
        )

        if self.option.is_call:
            return BlackScholesHedgingStrategy.call_delta(d)

        return BlackScholesHedgingStrategy.put_delta(d)

    @staticmethod
    def d1(S, K, r, sigma, t, T):
        t2m = T - t
        numerator = np.log(S / K) + (r + 0.5 * sigma**2) * t2m
        denominator = sigma * np.sqrt(t2m)

        with np.errstate(divide="ignore"):
            return numerator / denominator

    @staticmethod
    def d2(d1, sigma, t, T):
        t2m = T - t
        return d1 - sigma * np.sqrt(t2m)

    @staticmethod
    def call_price(S, K, r, t, T, d1, d2):
        return S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, r, t, T, d1, d2):
        return -S * norm.cdf(-d1) + K * np.exp(-r * (T - t)) * norm.cdf(-d2)

    @staticmethod
    def call_delta(d1):
        return norm.cdf(d1)

    @staticmethod
    def put_delta(d1):
        return norm.cdf(d1) - 1


class Pricer:
    def price(self, hedging_strategy: HedgingStrategy) -> float:
        S = hedging_strategy.stock.prices
        discount_factors = hedging_strategy.stock.discount_factors()
        delta = hedging_strategy.get_hedge_ratio()[:, :-1]
        txns = S * np.diff(delta, axis=1, prepend=0, append=0)

        payoff = hedging_strategy.option.payout()
        txns[:, -1] += payoff
        disounted_txns = discount_factors * txns
        costs = disounted_txns.sum(axis=1)

        return np.mean(costs)


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
        example_input = torch.rand(10000, 101, num_inputs)  # Random example input data
        return self(example_input).detach().numpy().squeeze(-1)


if __name__ == "__main__":
    S0 = 100
    K = 108
    T = 1
    num_steps = 100
    sigma = 0.2
    mu = 0.08
    r = 0.04
    n_path = 10_000

    t = np.linspace(0, T, num_steps + 1)

    if True:
        S = gbm(S0, r, sigma, t, n_path)
        d1 = _d1(S, K, r, sigma, t, T)
        d2 = _d2(d1, sigma, t, T)
        c = call_price(S, K, r, t, T, d1, d2)
        p = put_price(S, K, r, t, T, d1, d2)

        print(f"Call={c[0, 0]}")
        print(f"Put={p[0, 0]}")

        delta = call_delta(d1)[:, :-1]

        # Method 1
        # Cost of replication is the sum of each transaction plus the final payout
        discount_factors = np.exp(-r * t)
        txns = S * np.diff(delta, axis=1, prepend=0, append=0)
        call_payoff = np.maximum(S[:, -1] - K, 0)
        txns[:, -1] += call_payoff
        disounted_txns = discount_factors * txns
        costs = disounted_txns.sum(axis=1)
        print(f"Black-Scholes call price via replication = {np.mean(costs)}")

        txns = S * np.diff(delta - 1, axis=1, prepend=0, append=0)
        put_payoff = np.maximum(K - S[:, -1], 0)
        txns[:, -1] += put_payoff
        disounted_txns = discount_factors * txns
        costs = disounted_txns.sum(axis=1)
        print(f"Black-Scholes put  price via replication = {np.mean(costs)}")

        # Method 2
        # Cost of replication is the final payout less any gains from stock growth
        discounted_profits = np.diff(discount_factors * S, axis=1) * delta
        profits = np.exp(-r * T) * call_payoff - np.sum(discounted_profits, axis=1)
        print(f"Black-Scholes call price via replication = {np.mean(profits)}")

        discounted_profits = np.diff(discount_factors * S, axis=1) * (delta - 1)
        profits = np.exp(-r * T) * put_payoff - np.sum(discounted_profits, axis=1)
        print(f"Black-Scholes put  price via replication = {np.mean(profits)}")

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
    pricer.price(bs_strategy)

    # Example input data
    num_inputs = 4
    batch_size = 5
    example_input = torch.rand(batch_size, 2, num_inputs)  # Random example input data

    # Create an instance of the MLPHedgingStrategy
    mlp_strategy = MLPHedgingStrategy(
        option=call_option,
        num_inputs=num_inputs,
        num_hidden_layers=4,
        num_features=32,
        num_outputs=1,
    )

    mlp_strategy(example_input).squeeze(-1).detach().numpy().shape

    pricer = Pricer()
    pricer.price(mlp_strategy)
