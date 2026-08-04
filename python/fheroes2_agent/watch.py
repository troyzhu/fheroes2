"""Watch a policy play, decision by decision.

Usage: python3 -m fheroes2_agent.watch WORKER [--checkpoint P] [--attacker SPEC] [--defender SPEC]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .encoding import ENCODING_VERSION
from .env import BattleEnv
from .policy import BattlePolicy
from .render import describe_action, describe_army, draw_board, parse_army


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a battle policy play.")
    parser.add_argument("worker")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fixture", default="m1_tiny_melee")
    parser.add_argument("--side", default="attacker")
    parser.add_argument("--attacker", default=None)
    parser.add_argument("--defender", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--greedy", action="store_true", help="take the most likely legal action")
    parser.add_argument("--out", default=None, help="write the transcript here instead of stdout")
    args = parser.parse_args()

    # Names are friendlier than ids and the worker only understands ids.
    args.attacker = parse_army(args.attacker) if args.attacker else None
    args.defender = parse_army(args.defender) if args.defender else None

    model = BattlePolicy()
    label = "untrained policy"
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if state.get("encoding_version") != ENCODING_VERSION:
            raise ValueError(f"checkpoint encoding {state.get('encoding_version')} does not match {ENCODING_VERSION}")
        model.load_state_dict(state["state_dict"])
        label = args.checkpoint
    model.eval()

    env = BattleEnv(args.worker, fixture=args.fixture, side=args.side, attacker=args.attacker, defender=args.defender)
    out: list[str] = []

    def emit(text: str = "") -> None:
        out.append(text)

    emit(f"# Battle transcript")
    emit()
    emit(f"policy `{label}`, playing the {args.side}. "
         f"Attacker stacks are uppercase, defender lowercase, the stack on turn is in brackets.")
    if args.attacker or args.defender:
        emit()
        emit(f"- attacker: {describe_army(args.attacker) if args.attacker else 'fixture default'}")
        emit(f"- defender: {describe_army(args.defender) if args.defender else 'fixture default'}")
    emit()

    for episode in range(args.episodes):
        observation, mask = env.reset()
        emit(f"## Episode {episode + 1}")
        emit()
        step = 0
        while True:
            with torch.no_grad():
                logits, value = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(logits.argmax()) if args.greedy else int(distribution.sample())
                probability = float(torch.softmax(logits, dim=-1)[0, action])

            raw = env._pending["observation"]  # the record behind the encoded vector
            step += 1
            emit(f"### Decision {step}, round {raw['round']}")
            emit()
            emit("```")
            emit(draw_board(raw))
            emit("```")
            emit()
            emit(f"{int(mask.sum())} legal actions. Chose **{describe_action(action)}** "
                 f"(index {action}, probability {probability:.2f}, value estimate {float(value):+.2f}).")
            emit()

            result = env.step(action)
            if result.done:
                info = result.info
                emit(f"### Outcome")
                emit()
                emit(f"**{info['termination']}** after {info['rounds']} rounds and {step} decisions. "
                     f"Attacker left {info['attacker']['live_creatures']} creatures on "
                     f"{info['attacker']['live_stacks']} stacks, defender "
                     f"{info['defender']['live_creatures']} on {info['defender']['live_stacks']}. "
                     f"Reward {result.reward:+.3f}.")
                emit()
                break
            observation, mask = result.observation, result.mask

    env.close()
    text = "\n".join(out)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        print(f"wrote {args.out} ({len(out)} lines)")
    else:
        print(text)


if __name__ == "__main__":
    main()
