from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Generator

import numpy as np
from colorama import Fore, Style

UNKNOWN = 0
EXCLUDE = 1
INCLUDE = 2


class VectorType(Enum):
    ROW = 1
    COLUMN = 2


@dataclass
class KnapsackNumberLine:
    def __init__(self, numbers: np.ndarray, mask: np.ndarray) -> None:
        self.numbers: np.ndarray = numbers
        self.mask: np.ndarray = mask

        self._number_line, self._accessibles = self._build_accessible_numbers()

    def can_reach(self, target: int) -> bool:
        result = self._accessibles[self._number_line == target]
        return len(result) == 1 and result[0]

    def _build_accessible_numbers(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bound, upper_bound = self.get_bounds()
        number_line = np.arange(lower_bound, upper_bound + 1)
        accessibles = number_line == 0

        for num, m in zip(self.numbers, self.mask):
            if m == UNKNOWN:
                shifted = KnapsackNumberLine.shift(accessibles, num)
                accessibles |= shifted

        include_shift = np.sum(self.numbers[self.mask == INCLUDE])
        accessibles = KnapsackNumberLine.shift(accessibles, include_shift)

        return (number_line, accessibles)

    def get_bounds(self) -> tuple[int, int]:
        lower_bound = np.sum(self.numbers[self.numbers < 0])
        upper_bound = np.sum(self.numbers[self.numbers > 0])
        return lower_bound, upper_bound

    @staticmethod
    def shift(x: np.ndarray, n: int) -> np.ndarray:
        if n == 0:
            return x[:]

        if n > 0:
            return np.pad(x, (n, 0))[:-n]

        return np.pad(x, (0, -n))[-n:]


@dataclass
class BoardVector:
    numbers: np.ndarray
    mask: np.ndarray
    target: float
    vector_type: VectorType
    index: int

    def unknowns(self) -> Generator[int, None, None]:
        for i, m in enumerate(self.mask):
            if m == UNKNOWN:
                yield i

    def sum(self) -> int:
        included = self.mask != EXCLUDE
        return np.sum(self.numbers[included])

    def full_index(self, index: int) -> tuple[int, int]:
        if self.vector_type == VectorType.COLUMN:
            return index, self.index
        if self.vector_type == VectorType.ROW:
            return self.index, index
        raise ValueError(f"vector {self.vector_type} type unknown")

    def without(self, index: int) -> KnapsackNumberLine:
        mask = self.mask.copy()
        mask[index] = EXCLUDE
        return KnapsackNumberLine(self.numbers, mask)

    def include(self, index: int) -> None:
        self.mask[index] = INCLUDE

    def exclude(self, index: int) -> None:
        self.mask[index] = EXCLUDE


@dataclass
class Board:
    grid: np.ndarray
    mask: np.ndarray
    row_sums: np.ndarray
    col_sums: np.ndarray

    def unsolved_vectors(self) -> Generator[BoardVector, None, None]:
        yield from self.unsolved_rows()
        yield from self.unsolved_cols()

    def unsolved_rows(self) -> Generator[BoardVector, None, None]:
        for i in range(self.grid.shape[0]):
            row_mask = self.mask[i, :]
            if np.any(row_mask == UNKNOWN):
                grid, mask = self.grid[i, :].view(), self.mask[i, :].view()
                yield BoardVector(grid, mask, self.row_sums[i], VectorType.ROW, i)

    def unsolved_cols(self) -> Generator[BoardVector, None, None]:
        for j in range(self.grid.shape[1]):
            col_mask = self.mask[:, j]
            if np.any(col_mask == UNKNOWN):
                grid, mask = self.grid[:, j], self.mask[:, j]
                yield BoardVector(grid, mask, self.col_sums[j], VectorType.COLUMN, j)

    def is_solved(self) -> bool:
        included = self.mask != EXCLUDE
        col_sums = np.sum(self.grid * included, axis=0)
        row_sums = np.sum(self.grid * included, axis=1)

        return np.alltrue(col_sums == self.col_sums) and np.alltrue(row_sums == self.row_sums)

    def __copy__(self):
        return Board(self.grid.copy(), self.mask.copy(), self.row_sums.copy(), self.col_sums.copy())

    @classmethod
    def create(cls, size: int) -> "Board":
        rng = np.random.RandomState()

        if size > 8:
            value_range = np.concatenate((np.arange(-20, 0), np.arange(1, 21)))
            grid = np.random.choice(value_range, size=(size, size))
        else:
            grid = rng.randint(low=1, high=10, size=(size, size))

        true_mask = np.random.randint(low=1, high=3, size=(size, size))
        included = grid * (true_mask == INCLUDE)
        row_sums = np.sum(included, axis=1)
        col_sums = np.sum(included, axis=0)

        return cls(grid, grid * 0, row_sums, col_sums)


class BoardPrinter:
    def print(self, board: Board) -> None:
        rows, cols = board.grid.shape

        # Print header row
        col_nums = "    " + " ".join([f" {i+1:2}" for i in range(cols)])
        row_divider = "   |" + "-" * (cols * 4) + "|"
        print(col_nums)
        print(row_divider)

        # Print each row of the matrix
        for i in range(rows):
            row_items = []
            for j in range(cols):
                item = f"{board.grid[i][j]:3}"
                if board.mask[i][j] == 1:
                    item = f"{Fore.RED}{item}{Style.RESET_ALL}"
                elif board.mask[i][j] == 2:
                    item = f"{Fore.GREEN}{item}{Style.RESET_ALL}"
                row_items.append(item)

            row_string = " ".join(row_items)
            print(f"{i+1:2} |{row_string} |{board.row_sums[i]:3}")

        # Print the footer for the matrix
        col_sums = [f"{board.col_sums[j]:3}" for j in range(cols)]
        col_string = " ".join(col_sums)
        print(row_divider)
        print(f"   |{col_string} |")


class SumpleteSolver:
    def solve(self, board: Board) -> bool:
        is_updating = True
        while is_updating:
            is_updating = False
            for vector in board.unsolved_vectors():
                for index in vector.unknowns():
                    number_line = vector.without(index)
                    if not number_line.can_reach(vector.target):
                        vector.include(index)
                        is_updating = True
                    elif not number_line.can_reach(vector.target - vector.numbers[index]):
                        vector.exclude(index)
                        is_updating = True

                # If the target cannot be reached...
                if np.all(vector.mask != UNKNOWN) and vector.sum() != vector.target:
                    return False

        if board.is_solved():
            return True

        # Now we recurse...
        def backtrack(try_include: bool) -> bool:
            speculative_board = copy(board)
            vector = next(speculative_board.unsolved_vectors())
            index = next(vector.unknowns())

            if try_include:
                vector.include(index)
            else:
                vector.exclude(index)

            if self.solve(speculative_board):
                board.mask = speculative_board.mask
                return True

            return False

        return backtrack(True) or backtrack(False)


if __name__ == "__main__":
    n = 8
    board = Board.create(n)

    solver = SumpleteSolver()
    if solver.solve(board):
        printer = BoardPrinter()
        printer.print(board)
    else:
        print("failed to solve")
