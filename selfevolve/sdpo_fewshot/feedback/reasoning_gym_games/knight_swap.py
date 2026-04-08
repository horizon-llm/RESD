import json
from typing import Optional, Any


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score Knight Swap answer: validate moves and check if pieces are swapped."""
    if not isinstance(answer, str):
        return 0.0, "Empty or invalid answer."

    answer = answer.strip()
    if len(answer) == 0:
        return 0.0, "Empty or invalid answer."

    # Handle impossible puzzles
    if not entry["metadata"]["is_possible"]:
        if answer.lower() == "no":
            return 1.0, ""
        return 0.0, "This puzzle is impossible. The correct answer is 'no'."

    # Handle "No" answer for possible puzzles
    if answer.lower() == "no":
        solution = entry["metadata"].get("solution")
        if solution:
            formatted = [",".join(m) if isinstance(m, list) else m for m in solution]
            return 0.0, f"This puzzle is solvable in {len(solution)} moves. The correct solution is: {json.dumps(formatted)}"
        return 0.0, "This puzzle is solvable, but you answered 'no'."

    try:
        move_list = json.loads(answer)
        if not isinstance(move_list, list):
            return 0.0, "Answer must be a JSON list of moves."

        moves = []
        for idx, move_str in enumerate(move_list):
            parts = move_str.split(",")
            if len(parts) != 3:
                return 0.0, f"Move {idx + 1} has wrong format: '{move_str}'. Expected 'color,start,end'."
            color, start, end = parts
            if color not in ("w", "B"):
                return 0.0, f"Move {idx + 1}: invalid color '{color}'. Must be 'w' or 'B'."
            moves.append((color, start, end))

        board = entry["metadata"]["board"]
        pieces = dict(entry["metadata"]["pieces"])
        current_turn = entry["metadata"]["start_turn"]

        for idx, (color, start, end) in enumerate(moves):
            move_num = idx + 1
            if color != current_turn:
                return 0.0, f"Move {move_num}: wrong turn. Expected '{current_turn}' but got '{color}'."
            if start not in pieces or pieces[start] != color:
                return 0.0, f"Move {move_num}: no {color} piece at position {start}."
            if end not in board[start]:
                return 0.0, f"Move {move_num}: {start} -> {end} is not a valid knight move."
            if end in pieces and pieces[end] is not None:
                return 0.0, f"Move {move_num}: target position {end} is occupied by {pieces[end]}."

            pieces[end] = pieces[start]
            pieces[start] = None
            current_turn = "B" if current_turn == "w" else "w"

        # Check if solved
        white_positions = {pos for pos, piece in pieces.items() if piece == "w"}
        black_positions = {pos for pos, piece in pieces.items() if piece == "B"}
        initial_white = {pos for pos, piece in entry["metadata"]["pieces"].items() if piece == "w"}
        initial_black = {pos for pos, piece in entry["metadata"]["pieces"].items() if piece == "B"}

        if white_positions == initial_black and black_positions == initial_white:
            optimal_moves = len(entry["metadata"]["solution"])
            if len(moves) <= optimal_moves:
                return 1.0, ""
            else:
                return optimal_moves / len(moves), f"Solved but suboptimal: used {len(moves)} moves, optimal is {optimal_moves}."

        # Pieces moved but not fully swapped
        w_on_target = len(white_positions & initial_black)
        b_on_target = len(black_positions & initial_white)
        total_pieces = len(initial_white) + len(initial_black)
        on_target = w_on_target + b_on_target
        score = on_target / total_pieces if total_pieces > 0 else 0.0

        parts = []
        parts.append(f"Pieces not fully swapped after {len(moves)} moves. {on_target}/{total_pieces} pieces in target positions.")

        # White pieces: which are on target, which are misplaced
        w_misplaced = white_positions - initial_black
        if w_misplaced:
            targets_needed = initial_black - white_positions
            parts.append(f"White pieces not on target: {sorted(w_misplaced)}. They need to reach: {sorted(targets_needed)}.")

        # Black pieces: which are on target, which are misplaced
        b_misplaced = black_positions - initial_white
        if b_misplaced:
            targets_needed = initial_white - black_positions
            parts.append(f"Black pieces not on target: {sorted(b_misplaced)}. They need to reach: {sorted(targets_needed)}.")

        # Show current board state
        all_positions = sorted(set(list(pieces.keys()) + list(entry["metadata"]["pieces"].keys())))
        board_lines = []
        for pos in all_positions:
            occupant = pieces.get(pos)
            if occupant == "w":
                board_lines.append(f"  {pos}: white knight")
            elif occupant == "B":
                board_lines.append(f"  {pos}: BLACK knight")
            elif occupant is None:
                board_lines.append(f"  {pos}: empty")
        parts.append("Current board state:\n" + "\n".join(board_lines))

        # Show correct solution
        solution = entry["metadata"].get("solution")
        if solution:
            formatted = [",".join(m) if isinstance(m, list) else m for m in solution]
            parts.append(f"The correct solution ({len(solution)} moves): {json.dumps(formatted)}")

        feedback = "\n".join(parts)
        return score, feedback

    except json.JSONDecodeError as e:
        return 0.0, f"Failed to parse JSON: {e}"
    except Exception as e:
        return 0.0, f"Error validating moves: {e}"
