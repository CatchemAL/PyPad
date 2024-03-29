from __future__ import annotations

import asyncio
from asyncio import Event
from copy import copy
from dataclasses import dataclass, field
from math import sqrt
from typing import Awaitable, Generic, Self

import nest_asyncio
import numpy as np

from ..async_utils import asyncio_run
from ..states import State, TMove
from ..states.state import Policy
from .network import NeuralNetwork


@dataclass
class AlphaZeroMcts:
    neural_net: NeuralNetwork
    num_mcts_sims: int
    dirichlet_epsilon: float
    dirichlet_alpha: float
    discount_factor: float

    def policy(self, state: State[TMove], root_node: Node[TMove] | None = None) -> Policy[TMove]:
        return asyncio_run(self.policy_async(state, root_node))

    async def policy_async(
        self,
        state: State[TMove],
        root_node: Node[TMove] | None = None,
        cancellation_event: Event | None = None,
    ) -> Awaitable[Policy[TMove]]:
        root_node = await self.search_async(state, root_node, cancellation_event)
        root_q = 1 - root_node.q_value
        value = root_q * 2 - 1

        moves = [node.move for node in root_node.children]
        priors = np.array([node.visit_count for node in root_node.children], dtype=np.float32)
        priors /= priors.sum()

        policy_size = self.neural_net.game.config().action_size
        encoded_policy = np.zeros(policy_size)
        for move, prior in zip(moves, priors):
            loc = state.policy_loc(move)
            encoded_policy[loc] = prior

        return Policy(moves, priors, encoded_policy, value, root_node)

    def policies(
        self, states: list[State[TMove]], root_nodes: list[Node[TMove]] | None = None
    ) -> list[Policy]:
        root_nodes = self.search_parallel(states, root_nodes)

        policies: list[Policy] = []
        policy_shape = self.neural_net.game.config().action_size

        for state, root_node in zip(states, root_nodes):
            root_q = 1 - root_node.q_value
            value = root_q * 2 - 1

            moves = [node.move for node in root_node.children]
            new_priors = np.array([node.visit_count for node in root_node.children], dtype=np.float32)
            new_priors /= new_priors.sum()

            encoded_policy = np.zeros(policy_shape)
            for move, prior in zip(moves, new_priors):
                loc = state.policy_loc(move)
                encoded_policy[loc] = prior

            policy = Policy(moves, new_priors, encoded_policy, value, root_node)
            policies.append(policy)

        return policies

    def search(self, root_state: State[TMove], root_node: Node[TMove] | None = None) -> Node[TMove]:
        return asyncio_run(self.search_async(root_state, root_node))

    async def search_async(
        self,
        root_state: State[TMove],
        root_node: Node[TMove] | None = None,
        cancellation_event: Event | None = None,
    ) -> Awaitable[Node[TMove]]:
        root: Node[TMove] = root_node or Node(root_state.played_by)

        if root_node:
            # Add Dirchlet noise to the children if they are present
            ε = self.dirichlet_epsilon
            alpha = self.dirichlet_alpha
            noise = np.random.dirichlet([alpha] * len(root.children))
            for i, child_node in enumerate(root.children):
                child_node.prior = (1 - ε) * child_node.prior + ε * noise[i]

            root.parent = None  # free memory

        # check to bail more frequently if we are pondering - keep responsive
        poll_rate = 1 if cancellation_event else 5

        for n in range(self.num_mcts_sims):
            if n % poll_rate == 0:
                await asyncio.sleep(0)
            if cancellation_event and cancellation_event.is_set():
                break

            node = root
            state = copy(root_state)

            # === Selection ===
            while node.has_children:
                node = node.select_child()
                state.set_move(node.move)

            status = state.status()
            if status.is_in_progress:
                raw_policy, value = self.neural_net.predict(state)

                # Get the priors
                priors = np.zeros(len(status.legal_moves))
                for i, move in enumerate(status.legal_moves):
                    loc = state.policy_loc(move)
                    priors[i] = raw_policy[loc]
                priors /= np.sum(priors)

                if n == 0 and root_node is None:
                    ε = self.dirichlet_epsilon
                    alpha = self.dirichlet_alpha
                    noise = np.random.dirichlet([alpha] * len(priors))
                    priors = (1 - ε) * priors + ε * noise

                # === Expansion ===
                for move, prior in zip(status.legal_moves, priors):
                    child_played_by = 2 if state.played_by == 1 else 1
                    child_node = Node(child_played_by, parent=node, move=move, prior=prior)
                    node.children.append(child_node)

                # === Simulation ===
                # Here, the AlphaZero paper completely replaces the traditional rollout phase with
                # a value estimation from the neural net.
                # Negate because the net gives an estimation from player whose turn it is next,
                # rather than the player who has just moved
                value *= -1
            else:
                value = status.value

            # === Backpropagate ===
            node.backpropagate(value, self.discount_factor)

        return root

    def search_parallel(
        self, root_states: list[State[TMove]], root_nodes: list[Node[TMove]] | None = None
    ) -> list[Node]:
        roots = root_nodes or [Node(root_state.played_by) for root_state in root_states]
        if root_nodes:
            for root in roots:
                # Add Dirchlet noise to the children if they are present
                ε = self.dirichlet_epsilon
                alpha = self.dirichlet_alpha
                noise = np.random.dirichlet([alpha] * len(root.children))
                for i, child_node in enumerate(root.children):
                    child_node.prior = (1 - ε) * child_node.prior + ε * noise[i]

                root.parent = None  # free memory

        for n in range(self.num_mcts_sims):
            nodes = [root for root in roots]
            states: list[State[TMove]] = [copy(root_state) for root_state in root_states]

            # === Selection ===
            for j in range(len(nodes)):
                while nodes[j].has_children:
                    nodes[j] = nodes[j].select_child()
                    states[j].set_move(nodes[j].move)

            statuses = [state.status() for state in states]
            in_progress_idxs = [i for i, status in enumerate(statuses) if status.is_in_progress]
            finished_idxs = [i for i, status in enumerate(statuses) if not status.is_in_progress]
            num_in_progress = len(in_progress_idxs)
            values = np.zeros(len(statuses), dtype=np.float32)

            if num_in_progress > 0:
                in_progress_states = [states[idx] for idx in in_progress_idxs]
                raw_policies, predicted_values = self.neural_net.predict_parallel(in_progress_states)

                for i, idx in enumerate(in_progress_idxs):
                    state = states[idx]
                    status = statuses[idx]
                    raw_policy = raw_policies[i]
                    legal_moves = status.legal_moves

                    # Get the priors
                    priors = np.zeros(len(legal_moves))
                    for j, move in enumerate(legal_moves):
                        loc = state.policy_loc(move)
                        priors[j] = raw_policy[loc]
                    priors /= np.sum(priors)

                    if n == 0 and root_nodes is None:
                        ε = self.dirichlet_epsilon
                        alpha = self.dirichlet_alpha
                        noise = np.random.dirichlet([alpha] * len(priors))
                        priors = (1 - ε) * priors + ε * noise

                    # === Expansion ===
                    for move, prior in zip(legal_moves, priors):
                        child_played_by = 2 if state.played_by == 1 else 1
                        child_node = Node(child_played_by, nodes[idx], move, prior)
                        nodes[idx].children.append(child_node)

                # === Simulation ===
                # Here, the AlphaZero paper completely replaces the traditional rollout phase with
                # a value estimation from the neural net.
                # Negate because the net gives an estimation from player whose turn it is next,
                # rather than the player who has just moved
                values[in_progress_idxs] = -1 * predicted_values

            if finished_idxs:
                values[finished_idxs] = [statuses[idx].value for idx in finished_idxs]

            # === Backpropagate ===
            for i, node in enumerate(nodes):
                node.backpropagate(values[i], self.discount_factor)

        return roots


@dataclass(slots=True)
class Node(Generic[TMove]):
    played_by: int
    parent: Node[TMove] | None = None
    move: TMove | None = None
    prior: np.float32 = 1.0

    value_sum: float = field(default_factory=float, init=False)
    visit_count: int = field(default_factory=int, init=False)
    children: list[Node[TMove]] = field(default_factory=list, init=False)

    @property
    def has_children(self) -> bool:
        return bool(self.children)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            if self.parent is None:
                return 0

            FIRST_PLAY_URGENCY = 0.1  # 0.44
            q_from_parent = 1 - self.parent.q_value
            estimated_q_value = q_from_parent - FIRST_PLAY_URGENCY
            return max(estimated_q_value, 0)

        return (1 + self.value_sum / self.visit_count) / 2

    def update(self, outcome: float) -> None:
        self.visit_count += 1
        self.value_sum += outcome

    def backpropagate(self, outcome: float, discount_factor: float) -> None:
        self.update(outcome)
        if self.parent:
            self.parent.backpropagate(-outcome * discount_factor, discount_factor)

    def select_child(self) -> Self:
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self) -> float:
        c = 2  # todo AlphaZero sets to.... 2?
        exploration_param = sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + c * self.prior * exploration_param

    def __repr__(self):
        if self.parent:
            return f"Node(move={self.move}, Q={self.q_value:.1%}, prior={self.prior:.2%}, visit_count={self.visit_count}, UCB={self.ucb():.3})"
        return f"Node(move={self.move}, Q={self.q_value:.1%}, prior={self.prior:.2%}, visit_count={self.visit_count})"
