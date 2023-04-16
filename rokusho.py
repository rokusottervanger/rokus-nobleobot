import random
import sys
import numpy as np
from ..bot_control import Move


class Rokusho:

    def __init__(self):
        self.convertible_competitor_weight = 3
        self.zero_weight = 2
        self.resettable_weight = 1
        self.unconvertible_weight = 0

        self.move_coordinates = {Move.UP: [0, 1],
                                 Move.DOWN: [0, -1],
                                 Move.LEFT: [-1, 0],
                                 Move.RIGHT: [1, 0]}
                                #  Move.STAY: [0, 0]}

    def get_name(self):
        return "Rokusho"

    def get_contributor(self):
        return "Rokus"

    def get_neighbor_values(self, preference_grid, neighbors):
        move_values = {move: preference_grid[neighbor[1]][neighbor[0]] for move, neighbor in neighbors.items()}
        # print("Move values: \n{}".format(move_values))
        return move_values

    def determine_next_move(self, grid, enemies, game_info):
        np.set_printoptions(threshold=sys.maxsize, linewidth=200)

        preference_grid = self.determine_preference(grid)
        # print(preference_grid)

        # TODO(rokus): look a few steps ahead by summing cells along possible paths

        neighbors = self.get_neighbors(self.position, grid.shape)

        # self.print_neighbor_values(grid, neighbors)

        move_values = self.get_neighbor_values(preference_grid, neighbors)

        best_move_value = max(move_values.values())
        best_moves = [k for k, v in move_values.items() if v == best_move_value]

        return np.random.choice(best_moves, 1)[0]

    def print_neighbor_values(self, grid, neighbors):
        for move, idx in neighbors.items():
            print("Neighbor {}: {}".format(move, grid[idx[1]][idx[0]]))

    def get_neighbors(self, position, grid_size):
        neighbors = {}
        for move, side in self.move_coordinates.items():
            neighbor_coordinates = position + side
            if (neighbor_coordinates >= 0).all() and neighbor_coordinates[0] < grid_size[0] and neighbor_coordinates[1] < grid_size[1]:
                neighbors[move] = neighbor_coordinates
        return neighbors

    def determine_preference(self, grid):
        # Get a grid of ones for all zeros on the board
        zero_grid = (grid == 0).astype(int)

        # Run modulo on all competitor-owned cells on the board
        nonzeros_grid = (grid != 0).astype(int)
        mod_grid = np.mod((self.id - grid) * nonzeros_grid, 3)

        # Get the grid of all convertable competitors
        # TODO(@rokus): value higher scoring competitors more!!!
        convertible_competitor_grid = (mod_grid == 2).astype(int)

        # Get a grid of all resettable (return to zero) cells on the board
        resettable_grid = (mod_grid == 1).astype(int)

        unconvertible_grid = (mod_grid == 0).astype(int)

        preference_grid = convertible_competitor_grid * self.convertible_competitor_weight \
            + zero_grid * self.zero_weight \
            + resettable_grid * self.resettable_weight \
            + unconvertible_grid * self.unconvertible_weight

        return preference_grid
