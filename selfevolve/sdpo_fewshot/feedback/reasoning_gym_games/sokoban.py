"""Sokoban scoring: simulate moves on the puzzle grid and check if solved.

Embeds a minimal version of the sokoban game engine (Game, Player, Box, etc.)
from reasoning-gym's contrib/sokoban to avoid external dependency.
"""
from typing import Optional, Any

import numpy as np


# --- Minimal sokoban game engine ---

class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Box:
    def __init__(self, x, y, game=None):
        self.game = game
        self.x = x
        self.y = y

    def can_move(self, move):
        target_x, target_y = self.x + move[0], self.y + move[1]
        target = target_y, target_x
        curr = self.y, self.x
        target_elem = self.game.puzzle[target]
        if not isinstance(target_elem.obj, Box):
            curr_elem = self.game.puzzle[curr]
            self.y, self.x = target
            curr_elem.char = "-" if not curr_elem.ground else "X"
            curr_elem.obj = None
            target_elem.char = "@" if not target_elem.ground else "$"
            target_elem.obj = self
            return True
        return False


class Floor:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Goal(Floor):
    pass


class PuzzleElement:
    def __init__(self, char, obj=None, ground=None):
        self.char = char
        self.ground = ground
        self.obj = obj


class Player:
    def __init__(self, x, y, game):
        self.game = game
        self.x = x
        self.y = y

    def update(self, key=None):
        move = None
        if key:
            if key == "R":
                move = (1, 0)
            elif key == "L":
                move = (-1, 0)
            elif key == "U":
                move = (0, -1)
            elif key == "D":
                move = (0, 1)
        if move:
            target = self.y + move[1], self.x + move[0]
            target_elem = self.game.puzzle[target]
            if not (target_elem and target_elem.obj and isinstance(target_elem.obj, Obstacle)):
                is_box = isinstance(target_elem.obj, Box)
                if not is_box or (is_box and target_elem.obj.can_move(move)):
                    curr = self.y, self.x
                    curr_elem = self.game.puzzle[curr]
                    self.y, self.x = target
                    curr_elem.char = "-" if not curr_elem.ground else "X"
                    curr_elem.obj = None
                    target_elem.char = "*" if not target_elem.ground else "%"
                    target_elem.obj = self
                    return 1
        return 0


class Game:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.puzzle = np.empty((height, width), dtype=object)
        self.player = None
        self.puzzle_size = None
        self.pad_x = 0
        self.pad_y = 0

    def get_matrix(self):
        slice_x = slice(self.pad_x, self.pad_x + self.puzzle_size[1])
        slice_y = slice(self.pad_y, self.pad_y + self.puzzle_size[0])
        sliced = self.puzzle[slice_y, slice_x]
        matrix = np.empty(self.puzzle_size, dtype="<U1")
        for h in range(len(sliced)):
            for w in range(len(sliced[0])):
                matrix[h, w] = sliced[h, w].char
        return matrix

    def get_curr_state(self):
        return self.get_matrix().tobytes().decode("utf-8").replace("\x00", "")

    def load_puzzle_matrix(self, matrix):
        if isinstance(matrix, np.ndarray):
            data = matrix.tolist()
        else:
            data = matrix

        self.puzzle_size = (len(data), len(data[0]) if len(data) > 0 else 0)
        pad_x = (self.width - self.puzzle_size[1]) // 2
        pad_y = (self.height - self.puzzle_size[0]) // 2
        self.pad_x, self.pad_y = pad_x, pad_y

        for i, row in enumerate(data):
            for j, c in enumerate(row):
                new_elem = PuzzleElement(c)
                self.puzzle[i + pad_y, j + pad_x] = new_elem

                if c == "+":
                    new_elem.obj = Obstacle(x=j + pad_x, y=i + pad_y)
                elif c == "@":
                    new_elem.obj = Box(x=j + pad_x, y=i + pad_y, game=self)
                elif c == "*":
                    new_elem.obj = Player(x=j + pad_x, y=i + pad_y, game=self)
                    self.player = new_elem.obj
                elif c == "X":
                    new_elem.ground = Goal(x=j + pad_x, y=i + pad_y)
                elif c == "$":
                    new_elem.ground = Goal(x=j + pad_x, y=i + pad_y)
                    new_elem.obj = Box(x=j + pad_x, y=i + pad_y, game=self)
                elif c == "%":
                    new_elem.obj = Player(x=j + pad_x, y=i + pad_y, game=self)
                    new_elem.ground = Goal(x=j + pad_x, y=i + pad_y)
                    self.player = new_elem.obj

    def count_remaining_boxes(self):
        """Count boxes not yet on goals."""
        state = self.get_curr_state()
        return state.count("@")


def _is_solved(state):
    return "@" not in state


def _get_board_string(matrix):
    """Convert the game matrix to a human-readable board string."""
    rows = []
    for r in range(matrix.shape[0]):
        rows.append(" ".join(matrix[r]))
    return "\n".join(rows)


def _find_positions(matrix, chars):
    """Find all (row, col) positions of cells matching any char in chars."""
    positions = []
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if matrix[r, c] in chars:
                positions.append((r, c))
    return positions


def _detect_deadlocked_boxes(matrix):
    """Detect boxes that are stuck in corners and can never reach a goal.

    A box is deadlocked if it is against two perpendicular walls/obstacles,
    e.g. top-left corner, top-right corner, etc.
    """
    h, w = matrix.shape
    wall_chars = {"+", None}
    deadlocked = []
    # Only check boxes NOT on goals (char '@')
    for r, c in _find_positions(matrix, {"@"}):
        # Check if adjacent cells are walls or out of bounds
        up = matrix[r - 1, c] if r > 0 else None
        down = matrix[r + 1, c] if r < h - 1 else None
        left = matrix[r, c - 1] if c > 0 else None
        right = matrix[r, c + 1] if c < w - 1 else None

        blocked_up = up in wall_chars
        blocked_down = down in wall_chars
        blocked_left = left in wall_chars
        blocked_right = right in wall_chars

        # Corner deadlock: blocked on two perpendicular sides
        if ((blocked_up or blocked_down) and (blocked_left or blocked_right)):
            deadlocked.append((r, c))
    return deadlocked


# --- Scoring ---

def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score sokoban answer: simulate LRUD moves and check if all boxes are on goals."""
    if not isinstance(answer, str):
        return 0.0, "Empty or invalid answer."

    try:
        grid_list = [list(line) for line in entry["metadata"]["gamestr"].replace(" ", "").strip().split("\n")]
        matrix = np.array(grid_list)

        h, w = matrix.shape
        game = Game(height=h, width=w)
        game.load_puzzle_matrix(matrix)

        # Count initial boxes
        initial_state = game.get_curr_state()
        initial_boxes = initial_state.count("@")

        invalid_chars = [c for c in answer if c not in "LRUD"]
        if invalid_chars:
            return 0.0, f"Invalid move characters: {set(invalid_chars)}. Only L, R, U, D are allowed."

        for i, move in enumerate(answer):
            game.player.update(key=move)

        final_matrix = game.get_matrix()
        final_state = game.get_curr_state()
        if _is_solved(final_state):
            return 1.0, ""

        remaining = final_state.count("@")
        placed = initial_boxes - remaining
        score = placed / initial_boxes if initial_boxes > 0 else 0.0

        # Build detailed feedback
        parts = []
        parts.append(f"Puzzle not solved after {len(answer)} moves. {placed}/{initial_boxes} boxes on goals.")

        # Positions of misplaced boxes (not on goals)
        misplaced = _find_positions(final_matrix, {"@"})
        if misplaced:
            pos_str = ", ".join(f"(row={r}, col={c})" for r, c in misplaced)
            parts.append(f"Misplaced boxes at: {pos_str}.")

        # Positions of empty goals
        empty_goals = _find_positions(final_matrix, {"X"})
        if empty_goals:
            pos_str = ", ".join(f"(row={r}, col={c})" for r, c in empty_goals)
            parts.append(f"Empty goals at: {pos_str}.")

        # Deadlock detection
        deadlocked = _detect_deadlocked_boxes(final_matrix)
        if deadlocked:
            pos_str = ", ".join(f"(row={r}, col={c})" for r, c in deadlocked)
            parts.append(f"WARNING: {len(deadlocked)} box(es) are deadlocked (stuck in corners) at: {pos_str}. These can never be moved to a goal.")

        # Show resulting board state
        board_str = _get_board_string(final_matrix)
        parts.append(f"Board state after your moves:\n{board_str}")

        if entry.get("answer") is not None:
            parts.append(f"The correct solution is: {entry['answer']}")

        feedback = "\n".join(parts)
        return score, feedback
    except Exception as e:
        return 0.0, f"Failed to simulate moves: {e}"
