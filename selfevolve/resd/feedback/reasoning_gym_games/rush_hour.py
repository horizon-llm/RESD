import re
from typing import Optional, Any

BOARD_SIZE = 6
PRIMARY_ROW = 2
PRIMARY_SIZE = 2
MIN_PIECE_SIZE = 2
MAX_PIECE_SIZE = 3

BOARD_TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE
TARGET = PRIMARY_ROW * BOARD_SIZE + BOARD_SIZE - PRIMARY_SIZE
H = 1  # horizontal stride
V = BOARD_SIZE  # vertical stride


def _create_row_masks():
    row_masks = []
    for y in range(BOARD_SIZE):
        mask = 0
        for x in range(BOARD_SIZE):
            mask |= 1 << (y * BOARD_SIZE + x)
        row_masks.append(mask)
    return row_masks


def _create_column_masks():
    column_masks = []
    for x in range(BOARD_SIZE):
        mask = 0
        for y in range(BOARD_SIZE):
            mask |= 1 << (y * BOARD_SIZE + x)
        column_masks.append(mask)
    return column_masks


ROW_MASKS = _create_row_masks()
TOP_ROW = ROW_MASKS[0]
BOTTOM_ROW = ROW_MASKS[-1]
COLUMN_MASKS = _create_column_masks()
LEFT_COLUMN = COLUMN_MASKS[0]
RIGHT_COLUMN = COLUMN_MASKS[-1]


class Piece:
    def __init__(self, position, size, stride):
        self.position = position
        self.size = size
        self.stride = stride
        self.mask = 0
        p = position
        for _ in range(size):
            self.mask |= 1 << p
            p += stride

    @property
    def fixed(self):
        return self.size == 1

    def move(self, steps):
        d = self.stride * steps
        self.position += d
        if steps > 0:
            self.mask <<= d
        else:
            self.mask >>= -d


class Board:
    def __init__(self, desc):
        self._horz_mask = 0
        self._vert_mask = 0
        self._pieces = []

        if len(desc) != BOARD_TOTAL_CELLS:
            raise ValueError("board string is wrong length")

        positions = {}
        walls = []
        for i, label in enumerate(desc):
            if label == "o":
                continue
            if label == "x":
                walls.append(i)
                continue
            if label not in positions:
                positions[label] = []
            positions[label].append(i)

        labels = sorted(positions.keys())
        for label in labels:
            ps = positions[label]
            if len(ps) < MIN_PIECE_SIZE:
                raise ValueError("piece size < MinPieceSize")
            if len(ps) > MAX_PIECE_SIZE:
                raise ValueError("piece size > MaxPieceSize")
            stride = ps[1] - ps[0]
            if stride != H and stride != V:
                raise ValueError("invalid piece shape")
            for i in range(2, len(ps)):
                if ps[i] - ps[i - 1] != stride:
                    raise ValueError("invalid piece shape")
            self._add_piece(Piece(ps[0], len(ps), stride))

        for wall_pos in walls:
            self._add_piece(Piece(wall_pos, 1, 1))

    def _add_piece(self, piece):
        self._pieces.append(piece)
        if piece.stride == H:
            self._horz_mask |= piece.mask
        else:
            self._vert_mask |= piece.mask

    def _mask(self):
        return self._horz_mask | self._vert_mask

    def _do_move(self, i, steps):
        piece = self._pieces[i]
        if piece.stride == H:
            self._horz_mask &= ~piece.mask
            piece.move(steps)
            self._horz_mask |= piece.mask
        else:
            self._vert_mask &= ~piece.mask
            piece.move(steps)
            self._vert_mask |= piece.mask

    def move(self, target, direction):
        board_mask = self._mask()
        i = ord(target) - ord("A")
        if i < 0 or i > len(self._pieces):
            return False, f"Piece '{target}' does not exist on the board."

        piece = self._pieces[i]
        if piece.fixed:
            return False, f"Piece '{target}' is a wall/fixed piece and cannot move."

        moved = False
        for _ in range(abs(direction)):
            if piece.stride == H:
                if ((piece.mask & LEFT_COLUMN) == 0) and direction < 0:
                    mask = (piece.mask >> H) & ~piece.mask
                    if (board_mask & mask) == 0:
                        self._do_move(i, -1)
                        board_mask = self._mask()
                        moved = True
                        continue
                if ((piece.mask & RIGHT_COLUMN) == 0) and direction > 0:
                    mask = (piece.mask << H) & ~piece.mask
                    if (board_mask & mask) == 0:
                        self._do_move(i, 1)
                        board_mask = self._mask()
                        moved = True
                        continue
            else:
                if ((piece.mask & TOP_ROW) == 0) and direction < 0:
                    mask = (piece.mask >> V) & ~piece.mask
                    if (board_mask & mask) == 0:
                        self._do_move(i, -1)
                        board_mask = self._mask()
                        moved = True
                        continue
                if ((piece.mask & BOTTOM_ROW) == 0) and direction > 0:
                    mask = (piece.mask << V) & ~piece.mask
                    if (board_mask & mask) == 0:
                        self._do_move(i, 1)
                        board_mask = self._mask()
                        moved = True
                        continue
            # If we reach here, the step was blocked
            return False, f"Piece '{target}' is blocked and cannot move further in the requested direction."
        return True, ""

    def perform_moves(self, ops):
        pattern = r"([A-Z])([+-])(\d+)"
        matches = re.findall(pattern, ops)
        move_ops = [(chars, int(num) if sign == "+" else -int(num)) for chars, sign, num in matches]
        feedback_parts = []
        for idx, (target, direction) in enumerate(move_ops):
            ok, msg = self.move(target, direction)
            if not ok:
                feedback_parts.append(f"Move {idx + 1} ({target}{'+' if direction > 0 else ''}{direction}): {msg}")
        return feedback_parts

    @property
    def solved(self):
        return self._pieces[0].position == TARGET


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score Rush Hour answer by simulating moves on board."""
    if not isinstance(answer, str) or len(answer) == 0:
        return 0.0, "Empty or invalid answer."

    try:
        board = Board(entry["metadata"]["board_config"])
        move_errors = board.perform_moves(answer)

        if board.solved:
            if move_errors:
                return 1.0, "Puzzle solved, but some moves were blocked: " + " ".join(move_errors)
            return 1.0, ""

        feedback = "Puzzle not solved: primary piece (A) did not reach the exit."
        if move_errors:
            feedback += " Move issues: " + " ".join(move_errors)
        if entry.get("answer") is not None:
            feedback += f" The correct solution is: {entry['answer']}"
        return 0.01, feedback
    except Exception as e:
        return 0.0, f"Failed to simulate moves: {e}"
