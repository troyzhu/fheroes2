---
title: "Pokémon battles from scratch: the cheap self-play success and its teamplay ceiling"
type: paper
authors: Huang, Lee (CoG 2019); Angliss et al. (VGC-Bench, 2025)
year: 2019, 2025
quality: primary
urls:
  - https://ieee-cog.org/2019/papers/paper_175.pdf
  - https://arxiv.org/abs/2506.10326
  - https://github.com/cameronangliss/vgc-bench
tags: [reference, tabula-rasa, self-play, turn-based, prior-art]
---

# Pokémon from scratch, the success and the ceiling in one pair

Two results on the same game, read together on 2026-08-08 for the owner's tabula-rasa survey, bracket where pure self-play works in turn-based battling. Pokémon battles are turn-based with typed units and simultaneous move choice, adjacent to our domain rather than matched, but the pair is the cleanest controlled contrast the genre offers.

## Huang and Lee 2019, the strict success

Algorithm 1 initializes parameters randomly, and nothing anywhere touches human data: pure self-play PPO with GAE, both sides the same network, invalid actions masked and renormalized, a 1.33-million-parameter actor-critic over 128-wide entity embeddings. The reward is terminal win or loss plus two small event terms, a penalty of 0.0125 per own fainted unit and 0.0025 per super-effective hit, which the authors keep an order of magnitude below the outcome. The budget is the striking part: 500 iterations of 7,680 self-play matches, 3.84 million games in six days for about ninety-one dollars of cloud compute. Measured against fixed opponents over a thousand games each: 995 wins against random, 929 against max-damage, 829 against a type-aware max-damage, and 612 against pmariglia, a tree-search bot with a heuristic evaluation; on the live ladder it reached 1677 Glicko-1.

## VGC-Bench 2025, the matched negative at higher complexity

The same game's full six-versus-six team format, benchmarked properly: from-scratch self-play PPO loses cross-play to the simple heuristic player 48 to 52; fictitious play from scratch barely edges it; and initializing from behavior cloning flips the result decisively, 78 to 22 over the heuristic and past expert humans. The authors conclude the imitation start is essential at that complexity.

## Verdict for this project

The pair sharpens the genre law rather than contradicting it. Self-play from random initialization can beat search-backed heuristics when the per-turn decision is small and a near-terminal reward is reachable, and it stops working as branching and team structure grow, at which point imitation initialization is what restores it. Our battles sit between the two settings, and our program already stands on the imitation side of the line the second paper draws. The first paper's budget arithmetic transfers as encouragement: millions of self-play games are a weekend, not an infrastructure project, once the environment is fast.

## Related

- [[botbowl-competitions]], the same negative at extreme branching.
- [[vcmi-gym]], the genre-matched positive with shaping instead of team structure.
- [[../../rl/program-review]], the tabula-rasa standing this feeds.
