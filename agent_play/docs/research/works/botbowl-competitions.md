---
title: "botbowl and the Bot Bowl competitions — the genre's strongest tabula-rasa negative"
type: project
authors: Justesen, Moore, Uth, et al.
year: 2019-2023
quality: primary
urls:
  - https://github.com/njustesen/botbowl
  - https://sebastianrisi.com/wp-content/uploads/justesen_cog19a.pdf
  - https://arxiv.org/abs/2108.09478
tags: [reference, prior-art, tabula-rasa, turn-based, negative-result]
---

# botbowl, four competitions where from-scratch RL never beat a script

Blood Bowl is a turn-based board game of two teams of differentiated units, close kin to a battle here, and its open framework ran four annual bot competitions with both scripted and learned entries, which makes it the genre's best controlled record of what pure reinforcement learning could and could not do. Read 2026-08-08 for the owner's tabula-rasa survey.

## The measured record

The framework paper itself states the starting hopelessness: a random agent scored zero touchdowns in 350,000 matches, so the sparse natural reward is unreachable by exploration. The framework's own full-board tutorial trains vanilla A2C for 100 million steps, about a week on a desktop, to roughly 0.02 touchdowns per game against a random opponent; with the PPCG curriculum, difficulty auto-adjusted by widening the endzone, it reaches about 80 percent against random and is never claimed to beat anything scripted.

The competitions decide the question. Bot Bowl I: the scripted GrodBot won; the reinforcement entries lost. Bot Bowl II: scripted Minigrod won at 47 wins to 3 losses; the best learned bot, an A2C self-play entry, placed fourth, and an IMPALA-plus-LSTM entry finished last without a single win. Bot Bowl III was won by MimicBot, per its paper the first machine-learning agent to beat a scripted one there, and its recipe is the point: imitation of a scripted bot first, reinforcement fine-tuning after, hybrid decision-making on top. Bot Bowl IV: a scripted bot won again.

## Verdict for this project

Across four competitions, strictly-from-scratch reinforcement learning never beat a scripted bot; the one machine-learning win required an imitation bootstrap. The identified causes, enormous per-turn branching and a sparse score, are both milder in our battles, but the direction of the evidence matches our own tabula-rasa pilot, where search over a random prior distilled to a policy that evaluated at zero. The genre's record says the imitation start this project chose is not a convenience but the difference between the botbowl outcome and the [[vcmi-gym]] one.

## Related

- [[vcmi-gym]], the matched positive: from-scratch works there, with masking and dense shaping.
- [[../../rl/program-review]], where the tabula-rasa verdict this grounds is recorded.
