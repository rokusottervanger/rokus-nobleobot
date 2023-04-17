import random
import sys
import numpy as np
from scipy import signal
from ..bot_control import Move

def gkern(l, sig):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

class Rokusho:

    def __init__(self):
        self.convertible_competitor_weight = 3
        self.zero_weight = 2
        self.resettable_weight = 1
        self.unconvertible_weight = 0
        self.score_weight = 2

        self.move_coordinates = {Move.UP: np.array([0, 1],  dtype=np.int16),
                                 Move.RIGHT: np.array([1, 0],  dtype=np.int16),
                                 Move.LEFT: np.array([-1, 0], dtype=np.int16),
                                 Move.DOWN: np.array([0, -1], dtype=np.int16),
                                 Move.STAY: np.array([0, 0], dtype=np.int16)}

        np.set_printoptions(threshold=sys.maxsize, linewidth=200)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

        self.kernel = gkern(l=3, sig=1)
        self.max_blur_steps = 1

    def get_name(self):
        return "Rokusho"

    def get_contributor(self):
        return "Rokus"

    def get_neighbor_values(self, preference_grid, neighbors):
        move_values = {move: preference_grid[neighbor[1]][neighbor[0]] for move, neighbor in neighbors.items()}
        # print("Move values: \n{}".format(move_values))
        return move_values

    def determine_next_move(self, grid, enemies, game_info):
        self.game_info = game_info
        scoreboard = {enemy['id']: np.count_nonzero(grid == enemy['id']) for enemy in enemies}
        neighbors = self.get_neighbors(self.position, grid.shape)

        preference_grid = self.determine_preference(grid, scoreboard)
        best_moves = self.get_best_moves(preference_grid, neighbors)

        self.print_neighbor_values(preference_grid, neighbors)

        # Throw away neighbors that are not in best moves
        neighbors = {move: neighbor for move, neighbor in neighbors.items() if move in best_moves}

        steps = 0
        while len(best_moves) > 1 and steps < self.max_blur_steps:
            blurred_grid = signal.convolve2d(preference_grid, self.kernel, mode='same')
            self.print_neighbor_values(blurred_grid, neighbors)
            best_moves = self.get_best_moves(blurred_grid, neighbors)

            # Throw away neighbors that are not in best moves
            neighbors = {move: neighbor for move, neighbor in neighbors.items() if move in best_moves}
            steps += 1

        print("\n")

        return np.random.choice(best_moves, 1)[0]

    def get_best_moves(self, preference_grid, neighbors):
        move_values = self.get_neighbor_values(preference_grid, neighbors)
        best_move_value = max(move_values.values())
        best_moves = [k for k, v in move_values.items() if v == best_move_value]
        return best_moves

    def print_neighbor_values(self, grid, neighbors):
        print("Neighbor values:")
        for move, idx in neighbors.items():
            print("Neighbor {}: {}".format(move, grid[idx[1]][idx[0]]))

    def get_neighbors(self, position, grid_size):
        neighbors = {}
        for move, side in self.move_coordinates.items():
            neighbor_coordinates = position + side
            if (neighbor_coordinates >= 0).all() and neighbor_coordinates[0] < grid_size[0] and neighbor_coordinates[1] < grid_size[1]:
                neighbors[move] = neighbor_coordinates
        return neighbors

    def determine_preference(self, grid, scoreboard):
        # Get a grid of ones for all zeros on the board
        zero_grid = (grid == 0).astype(int)

        # Run modulo on all competitor-owned cells on the board
        nonzeros_grid = (grid != 0).astype(int)
        mod_grid = np.mod((self.id - grid) * nonzeros_grid, 3)

        # Get a grid of all resettable (return to zero) cells on the board
        resettable_grid = (mod_grid == 1).astype(int)

        # Get the grid of all convertable competitors
        convertible_competitor_grid = (mod_grid == 2).astype(int)

        # Calculate enemy scores normalized to the highest score so far
        max_score = max(max(scoreboard.values()), 1)
        normalized_enemy_scores = {k: v/max_score for k, v in scoreboard.items() if k is not self.id}
        print("Normalized enemy scores: \n{}".format(normalized_enemy_scores))

        # Substitute the scores in the respective grid cells
        enemy_score_grid = np.zeros_like(grid, dtype=float)
        for enemy, score in normalized_enemy_scores.items():
            enemy_score_grid += (grid == enemy).astype(float) * score

        game_progress = self.game_info.number_of_rounds / self.game_info.current_round
        game_progress_multiplier = (game_progress-0.5) * 2 if game_progress > 0.5 else 0
        enemy_score_grid *= (convertible_competitor_grid * self.score_weight + resettable_grid) \
            * game_progress_multiplier

        unconvertible_grid = (mod_grid == 0).astype(int) * nonzeros_grid

        preference_grid = convertible_competitor_grid * self.convertible_competitor_weight \
            + zero_grid * self.zero_weight \
            + resettable_grid * self.resettable_weight \
            + unconvertible_grid * self.unconvertible_weight \
            + enemy_score_grid * self.score_weight

        return preference_grid
