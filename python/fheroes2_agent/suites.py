"""The evaluation suites, and which chair each is played from.

Lifted out of `agent_play/experiments/validation_battery.py` on 2026-08-10 so the battery, the
search harnesses and anything measuring against the built-in AI baseline share one definition. They
had been reaching for it across a `sys.path` insert, which is how `search_agent_battery.py` came to
import the battery module purely for two names.

The suites are fixed collections of matchups, each probing something the others do not. Held-out
matchups were never trained on; the mirror suites give both sides the identical army under identical
commanders so neither is favoured by construction; the real-map suite lifts armies from shipped
maps; the stress suites push at hordes, wide units and commanders. A number is only comparable
across runs because these lists are fixed, so treat edits to them as breaking changes and say so in
the report that first uses the new list.

The built-in AI's own rate on every one of these is vendored in the baseline report, and any claim
about beating the engine is a comparison against that column.
"""

from __future__ import annotations

import json
import pathlib
import random

from .scenarios import Matchup, sample_budget_matchup

#: Vendored run reports the suites are built from. Resolved relative to this file rather than to a
#: script, which is what the move out of `agent_play/experiments/` changed: the old paths counted
#: directory levels from the script's location.
_ARCHIVE = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files"
POOL = _ARCHIVE / "2026-08-05-run-reports" / "pool_value.json"
REAL_MAPS_MANIFEST = _ARCHIVE / "2026-08-07-run-reports" / "real_map_fights.json"

THUNK_ARMY = "11:1,11:1,11:1,10:2,9:2"
THUNK_HERO = "13:12"


def thunk_split(total: int) -> str:
    first = total // 3 + (1 if total % 3 else 0)
    return f"1:{first},1:{total // 3},1:{total - first - total // 3}"


#: Which chair each suite is played from; attacker unless listed.
SUITE_SIDE = {"held_out_as_defender": "defender", "mirrors_defender": "defender"}

def real_map_suite() -> list:
    """Real opening fights harvested from the shipped maps, membership frozen by the vendored
    manifest so the column is stable across runs; real_map_fights.py regenerates it."""
    entries = json.loads(REAL_MAPS_MANIFEST.read_text())["fights"]
    return [Matchup(e["attacker"], e["defender"], attacker_hero=e["attacker_hero"], allow_wide=True)
            for e in entries]


def build_suites(fresh_count: int) -> dict[str, list[Matchup]]:
    suites: dict[str, list[Matchup]] = {}

    # 1. Fresh samples: the generator's raw distribution, seed never used anywhere else.
    rng = random.Random(20260805)
    suites["fresh_sampled"] = [sample_budget_matchup(rng) for _ in range(fresh_count)]

    # 2. Held-out pool, the split every result today reported.
    entries = json.loads(POOL.read_text())["matchups"]
    suites["held_out_pool"] = [Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"),
                                       defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")))
                               for e in entries[40:60]]

    # 3. Stress: hordes beyond every recorded count (training and its supplement stop at 1,000).
    suites["stress_hordes"] = [
        Matchup(THUNK_ARMY, thunk_split(total), attacker_hero=THUNK_HERO, allow_wide=True)
        for total in (1500, 2000, 3000)
    ] + [
        Matchup("9:4,10:6,6:12", thunk_split(total), attacker_hero="10:10", allow_wide=True)
        for total in (1500, 2500)
    ]

    # 4. Stress: armies of only two-cell creatures (the whole wide_v1 roster: Cavalry, Champion,
    # Wolf, Unicorn, Centaur, Boar, Nomad, Medusa), a composition the samplers rarely draw.
    wide_armies = [("9:3,28:4,62:3", "8:6,59:5,15:8"), ("15:12,30:9,40:7", "9:2,62:4,28:3"), ("8:8,9:2", "40:10,30:8,15:6")]
    suites["stress_wide_only"] = [Matchup(a, d, attacker_hero="8:8", defender_hero="8:8", allow_wide=True)
                                  for a, d in wide_armies]

    # 5. Stress: commander extremes on one mid pool matchup, stats far outside the sampled range.
    base = entries[45]
    suites["stress_commanders"] = [
        Matchup(base["attacker"], base["defender"], attacker_hero=hero, defender_hero=base.get("defender_hero"),
                allow_wide=bool(base.get("allow_wide")))
        for hero in ("0:0", "30:30", "99:0", "0:99")
    ]

    # 6. The standing ladder, untouched by every training set, plus rungs beyond the real fight.
    suites["thunk_ladder"] = [Matchup(THUNK_ARMY, thunk_split(total), attacker_hero=THUNK_HERO, allow_wide=True)
                              for total in (500, 700, 850, 1000)]

    # 7. Side coverage: the held-out pool commanded from the defender's chair, and mirror armies
    # from both chairs, since the side-swap measurements showed play quality is side-dependent.
    suites["held_out_as_defender"] = list(suites["held_out_pool"])
    mirrors = ["9:2,11:2,6:12,1:30", "62:3,30:6,15:10", "13:3,48:12,12:20", "10:4,7:8", "28:3,40:8,2:15", "51:4,50:4,12:16"]
    suites["mirrors_attacker"] = [Matchup(a, a, attacker_hero="10:10", defender_hero="10:10", allow_wide=True)
                                  for a in mirrors]
    suites["mirrors_defender"] = list(suites["mirrors_attacker"])

    suites["real_maps"] = real_map_suite()
    return suites


# Which side each suite is played from; attacker unless listed.
SUITE_SIDE = {"held_out_as_defender": "defender", "mirrors_defender": "defender"}
