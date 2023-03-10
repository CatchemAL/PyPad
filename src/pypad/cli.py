import numpy as np

from .kaggle_sandbox import run_kaggle
from .mcts import mcts, tictactoe
from .mcts_az import mcts_az
from .mnist.mnist_loader import load_data_wrapper
from .mnist.mnist_mlp import mlp_run
from .mnist.mnist_svm import svm_baseline
from .mnist.mnist_torch import run_torch
from .solver import Solver
from .state import ConnectX


def main() -> None:
    mcts_az()
    tictactoe()
    mcts()
    return
    run_kaggle()

    ROWS, COLS = 6, 7
    moves = [1, 1, 2, 2, 3, 3]
    # 12211221122137477577675665566556
    sequence = "1,2,2,1,1,2,2,1,1,2,2,1,3,7,4,7,7,5,7,7,6,7,5,6,6,5,5,6,6,5,5,6".split(",")
    moves = [int(s) for s in sequence]
    board = ConnectX.create(ROWS, COLS, moves)
    solver = Solver()
    solver.minimax(board, -np.inf, np.inf)

    run_torch()
    return
    mlp_run()
    svm_baseline()

    print("Hello world")
