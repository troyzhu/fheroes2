---
title: "Lux AI seasons 1 and 2 — from-scratch self-play wins once, rules win the rematch"
type: project
authors: Pressman (Toad Brigade); ry_andy_; flg; RoboEden (Jux)
year: 2021-2023
quality: primary
urls:
  - https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
  - https://github.com/ryandy/Lux-S2-public
  - https://www.kaggle.com/competitions/lux-ai-season-2/discussion/406702
  - https://github.com/RoboEden/jux
tags: [reference, tabula-rasa, self-play, prior-art, curriculum]
---

# Lux AI, the from-scratch win and the rules rematch

Two seasons of an open unit-economy strategy competition, read 2026-08-08 for the owner's tabula-rasa survey, and together they carry both halves of the genre's answer.

## Season 1: Toad Brigade, strict from-scratch, first place

The winner used no demonstrations and no scripted incumbent, because none existed: pure self-play IMPALA with UPGO and TD($\lambda$), a fully convolutional residual network with squeeze-excitation blocks and per-cell action heads, trained on one personal dual-GPU machine over the competition months. The recipe's load-bearing structure is a two-stage anneal with a distillation bridge: a small network first learns on a shaped reward over cities, units, research and fuel, then successively larger networks train on the sparse win-loss signal alone with the previous network held frozen as a teacher through a KL term, self-distillation carrying the shaped-era competence into the sparse era. The imitation cluster came after: several top-ten entries were supervised copies of the winner's public replays, imitation as the cheap follower strategy rather than the winning one.

## Season 2: rules beat reinforcement, even with a GPU engine on offer

The rematch season had heavier engineering everywhere and a stateful rule-based TypeScript bot won both the main season and the NeurIPS reinforcement-scaling stage; the best pure reinforcement entry placed fourth, a from-scratch self-play agent notable mostly for its inference engineering, and the JAX engine built precisely to enable large-scale training did not convert into a winning learned agent. Scale of environment throughput was not the binding constraint; the game had grown a decisive hand-designed strategy layer.

## Verdict for this project

Season 1 is the cleanest existence proof that shaped-then-sparse annealing with a frozen-teacher KL bridge lets pure self-play reach first place on hobby hardware when no strong scripted opponent exists, and season 2 is the sober counterweight: when a strong hand-built policy exists, from-scratch learning did not beat it there. Our setting is season-2-shaped, a strong scripted incumbent exists, which is the ecology where the successful projects train against the incumbent directly rather than through pure self-play, exactly the anchor-opponent structure our pool already has. The annealing-with-teacher-KL pattern is the transferable design, close kin to our early-stop-then-reinforce staging with the anchor as reference.

## Related

- [[gym-microrts]], the matched ecology: a strong scripted incumbent, trained against directly.
- [[alphastar-unplugged]], the league counterpart at industrial scale.
- [[../../rl/program-review]], where the tabula-rasa standing lives.
