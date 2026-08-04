"""Draw a battle as text, so a human can watch what a policy is doing.

The environment reads engine state directly and renders nothing, which is the point of the
design. That leaves no way to see a battle, and a policy that is quietly doing something absurd
looks exactly like one that is playing well when the only output is a win rate. This draws the
board from the same observation the policy receives, so what is shown is what the policy saw.

The board is 11 wide and 9 tall with hexagonal adjacency and row offsets, so odd rows are drawn
half a cell to the right, matching how the game itself lays them out.
"""

from __future__ import annotations

from .encoding import BOARD_HEIGHT, BOARD_WIDTH

MONSTER_NAMES = {
    1: "Peasant", 2: "Archer", 3: "Ranger", 4: "Pikeman", 5: "VetPikeman",
    6: "Swordsman", 7: "MstSwordsman", 8: "Cavalry", 9: "Champion", 10: "Paladin",
    11: "Crusader", 12: "Goblin", 13: "Orc", 14: "OrcChief", 15: "Wolf",
    16: "Ogre", 17: "OgreLord", 18: "Troll", 19: "WarTroll", 20: "Cyclops",
}


def monster_name(monster_id: int) -> str:
    return MONSTER_NAMES.get(monster_id, f"id{monster_id}")


def draw_board(observation: dict, chosen: str | None = None) -> str:
    """One board, with a legend. Attacker stacks are uppercase, defender lowercase."""
    occupants: dict[int, dict] = {}
    for unit in observation["units"]:
        if unit["head_cell"] >= 0:
            occupants[unit["head_cell"]] = unit

    # Stable single-letter labels: A, B, C for the attacker and a, b, c for the defender.
    labels: dict[int, str] = {}
    a_next, d_next = 0, 0
    for unit in sorted(observation["units"], key=lambda u: u["uid"]):
        if unit["side"] == "attacker":
            labels[unit["uid"]] = chr(ord("A") + a_next)
            a_next += 1
        else:
            labels[unit["uid"]] = chr(ord("a") + d_next)
            d_next += 1

    lines = []
    header = "     " + "".join(f"{c:>4d}" for c in range(BOARD_WIDTH))
    lines.append(header)
    for row in range(BOARD_HEIGHT):
        indent = "  " if row % 2 else ""
        cells = []
        for column in range(BOARD_WIDTH):
            cell = row * BOARD_WIDTH + column
            unit = occupants.get(cell)
            if unit is None:
                cells.append("  . ")
            else:
                mark = labels[unit["uid"]]
                cells.append(f" [{mark}]" if unit["active"] else f"  {mark} ")
        lines.append(f"{row:>3}  {indent}" + "".join(cells))

    lines.append("")
    for unit in sorted(observation["units"], key=lambda u: (u["side"] != "attacker", u["uid"])):
        mark = labels[unit["uid"]]
        turn = " <- on turn" if unit["active"] else ""
        shots = f", {unit['shots']} shots" if unit["archer"] else ""
        lines.append(
            f"  {mark}  {monster_name(unit['monster_id']):<12} {unit['side']:<8} "
            f"cell {unit['head_cell']:>2}  {unit['count']:>3}/{unit['initial_count']:<3} creatures  "
            f"{unit['hit_points']:>4} hp{shots}{turn}"
        )
    if chosen:
        lines.append(f"\n  chose: {chosen}")
    return "\n".join(lines)


def describe_action(index: int) -> str:
    """Canonical index back into readable form, mirroring ADR 0002's layout."""
    if index == 0:
        return "SKIP"
    if index < 100:
        return f"MOVE to cell {index - 1}"
    if index < 199:
        return f"SHOOT the stack on cell {index - 100}"
    offset = index - 199
    cell, direction = divmod(offset, 6)
    names = ["top-left", "top-right", "right", "bottom-right", "bottom-left", "left"]
    return f"MELEE cell {cell} from the {names[direction]}"
