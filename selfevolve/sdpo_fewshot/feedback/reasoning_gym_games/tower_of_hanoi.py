import re
from typing import Optional, Any


def _parse_move(move: str) -> tuple[int, int, int]:
    """Parse a move string: 'Move disk X from Peg Y to Peg Z'."""
    pattern = r"Move disk (\d+) from Peg (\d+) to Peg (\d+)"
    match = re.search(pattern, move)
    if not match:
        raise ValueError(f"Unexpected move format: '{move}'")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _validate_move(pegs_state: dict[int, list[int]], move: str) -> tuple[bool, str]:
    """Validate that a move adheres to the Tower of Hanoi rules."""
    try:
        parts = move.split()
        if len(parts) != 9:
            return False, f"Invalid move format (expected 9 tokens, got {len(parts)}): '{move}'"

        try:
            disk = int(parts[2])
            from_peg = int(parts[5])
            to_peg = int(parts[8])
        except ValueError:
            return False, f"Cannot parse disk/peg numbers from: '{move}'"

        if not pegs_state[from_peg] or pegs_state[from_peg][-1] != disk:
            top_disk = pegs_state[from_peg][-1] if pegs_state[from_peg] else "empty"
            return False, f"Disk {disk} is not on top of Peg {from_peg} (top is {top_disk})."

        if pegs_state[to_peg] and pegs_state[to_peg][-1] < disk:
            return False, f"Cannot place disk {disk} on top of smaller disk {pegs_state[to_peg][-1]} on Peg {to_peg}."

        return True, ""
    except Exception as e:
        return False, f"Move validation error: {e}"


def _build_peg_state_feedback(peg_state, num_pegs, num_disks, target_peg):
    """Build detailed feedback lines describing the current peg state."""
    parts = []

    # Show current state of each peg
    for peg in range(1, num_pegs + 1):
        disks = peg_state[peg]
        marker = " (target)" if peg == target_peg else ""
        if disks:
            parts.append(f"  Peg {peg}{marker}: disks {disks} (bottom to top)")
        else:
            parts.append(f"  Peg {peg}{marker}: empty")

    # Check ordering on target peg
    target_disks = peg_state[target_peg]
    if target_disks and target_disks != sorted(target_disks, reverse=True):
        parts.append(f"WARNING: Disks on target Peg {target_peg} are not in valid order: {target_disks}.")

    # Which disks are missing from target peg?
    missing = sorted(set(range(1, num_disks + 1)) - set(target_disks))
    if missing:
        missing_locations = []
        for d in missing:
            for peg in range(1, num_pegs + 1):
                if d in peg_state[peg]:
                    missing_locations.append(f"disk {d} is on Peg {peg}")
                    break
        parts.append(f"Disks not on target: {', '.join(missing_locations)}.")

    return parts


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score Tower of Hanoi answer: validate moves and simulate peg state."""
    if not isinstance(answer, str) or len(answer) == 0:
        return 0.0, "Empty or invalid answer."

    moves = [line.strip() for line in answer.strip().splitlines() if line.strip()]

    metadata = entry["metadata"]
    num_disks = metadata["num_disks"]
    num_pegs = metadata["num_pegs"]
    start_peg = metadata["start_peg"]
    target_peg = metadata["target_peg"]

    peg_state = {peg: [] for peg in range(1, num_pegs + 1)}
    for disk in range(num_disks, 0, -1):
        peg_state[start_peg].append(disk)

    for i, move in enumerate(moves):
        try:
            disk, from_peg, to_peg = _parse_move(move)
        except Exception as e:
            return 0.0, f"Move {i + 1} parse error: {e}"

        valid, reason = _validate_move(peg_state, move)
        if not valid:
            on_target = len(peg_state[target_peg])
            score = on_target / num_disks if num_disks > 0 else 0.0

            parts = []
            parts.append(f"Move {i + 1} is illegal. {reason}")
            parts.append(f"{on_target}/{num_disks} disks on target Peg {target_peg} at time of failure.")
            parts.extend(_build_peg_state_feedback(peg_state, num_pegs, num_disks, target_peg))
            parts.append(f"The correct solution is:\n{entry['answer']}")
            feedback = "\n".join(parts)
            return score, feedback

        peg_state[from_peg].pop()
        peg_state[to_peg].append(disk)

    expected_final = list(range(num_disks, 0, -1))
    solved = peg_state[target_peg] == expected_final
    if not solved:
        on_target = len(peg_state[target_peg])
        score = on_target / num_disks if num_disks > 0 else 0.0

        parts = []
        parts.append(f"Puzzle not solved after {len(moves)} moves. {on_target}/{num_disks} disks on target Peg {target_peg}.")
        parts.extend(_build_peg_state_feedback(peg_state, num_pegs, num_disks, target_peg))
        parts.append(f"The correct solution is:\n{entry['answer']}")
        feedback = "\n".join(parts)
        return score, feedback

    optimal_moves = metadata.get("solution_length", len(moves))
    user_moves = len(moves)
    if user_moves <= optimal_moves:
        return 1.0, ""
    else:
        return optimal_moves / user_moves, f"Solved but suboptimal: used {user_moves} moves, optimal is {optimal_moves}."
