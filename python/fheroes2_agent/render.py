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

# Generated from the engine's own capability audit (--capability-audit), so a name here always
# matches what the engine calls that creature. Hand-maintaining this drifted once already:
# it covered ids 1 to 20 and silently rejected every creature above them.
MONSTER_NAMES = {1: "Peasant", 2: "Archer", 3: "Ranger", 4: "Pikeman", 5: "Veteran Pikeman", 6: "Swordsman", 7: "Master Swordsman", 8: "Cavalry", 9: "Champion", 10: "Paladin", 11: "Crusader", 12: "Goblin", 13: "Orc", 14: "Orc Chief", 15: "Wolf", 16: "Ogre", 17: "Ogre Lord", 18: "Troll", 19: "War Troll", 20: "Cyclops", 21: "Sprite", 22: "Dwarf", 23: "Battle Dwarf", 24: "Elf", 25: "Grand Elf", 26: "Druid", 27: "Greater Druid", 28: "Unicorn", 29: "Phoenix", 30: "Centaur", 31: "Gargoyle", 32: "Griffin", 33: "Minotaur", 34: "Minotaur King", 35: "Hydra", 36: "Green Dragon", 37: "Red Dragon", 38: "Black Dragon", 39: "Halfling", 40: "Boar", 41: "Iron Golem", 42: "Steel Golem", 43: "Roc", 44: "Mage", 45: "Archmage", 46: "Giant", 47: "Titan", 48: "Skeleton", 49: "Zombie", 50: "Mutant Zombie", 51: "Mummy", 52: "Royal Mummy", 53: "Vampire", 54: "Vampire Lord", 55: "Lich", 56: "Power Lich", 57: "Bone Dragon", 58: "Rogue", 59: "Nomad", 60: "Ghost", 61: "Genie", 62: "Medusa", 63: "Earth Elemental", 64: "Air Elemental", 65: "Fire Elemental", 66: "Water Elemental"}


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


# Reverse of MONSTER_NAMES, so armies can be written as "archer:6,peasant:20" instead of "2:6,1:20".
def _key(name: str) -> str:
    """Normalize a creature name for lookup: case, spaces and punctuation all ignored.

    The engine spells them "Veteran Pikeman" and a command line is easier to type as
    "veteranpikeman", so both resolve. Matching the engine string exactly was tried and broke
    every army spec the moment the name table was regenerated from the engine.
    """
    return "".join(c for c in name.lower() if c.isalnum())


NAME_TO_ID = {_key(name): monster_id for monster_id, name in MONSTER_NAMES.items()}

# Short forms that are not just whitespace variants.
NAME_TO_ID.update({
    "vetpikeman": 5, "mstswordsman": 7, "grandelf": 25, "greaterdruid": 27,
    "minotaurking": 34, "vampirelord": 54, "powerlich": 56, "bonedragon": 57,
    "airelem": 64, "earthelem": 63, "fireelem": 65, "waterelem": 66,
})


def parse_army(spec: str) -> str:
    """Accept names or ids and return the id form the worker expects.

    "pikeman:20,archer:10" and "4:20,2:10" mean the same thing. The worker parses ids only,
    because a name table in C++ would duplicate the engine's own translated strings.
    """
    parts = []
    for item in spec.split(","):
        name, _, count = item.strip().partition(":")
        key = _key(name)
        monster_id = NAME_TO_ID.get(key)
        if monster_id is None:
            if not key.isdigit():
                known = ", ".join(sorted(NAME_TO_ID))
                raise ValueError(f"unknown creature {name!r}; known names are {known}")
            monster_id = int(key)
        parts.append(f"{monster_id}:{int(count)}")
    return ",".join(parts)


def describe_army(spec: str) -> str:
    """The id form back into something readable, for a transcript header."""
    out = []
    for item in spec.split(","):
        monster_id, _, count = item.partition(":")
        out.append(f"{count} {monster_name(int(monster_id))}")
    return ", ".join(out)
