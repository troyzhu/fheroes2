#!/usr/bin/env python3
"""Render a captured replay through the real engine and assemble a video.

`capture_replay.py` records what the policy did; this makes it watchable with the game's own
renderer. The pipeline is verification-first: the recording is replayed headless and must
reproduce exactly (every action legal at its decision point, all consumed, same termination),
then replayed rendered, and the two terminal state digests must be identical, which proves the
video shows precisely the battle the agent played. Frames come from the replay tool's frame
dumper (installed on the display's generic render observer), with real animation timing in the
manifest, and ffmpeg reassembles them at that timing.

The rendered pass opens a real game window for the duration of the battle; the machine needs a
display session, the game assets (found through the usual root dirs), and ffmpeg.

Usage:
    ./render_replay.py REPLAY_JSON OUT_MP4 [--speed 10] [--scale 2] [--hold-last 2.0]
                       [--allow-wide] [--keep-frames]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPLAY_BIN = REPO_ROOT / "src" / "agent_replay" / "fheroes2_agent_replay"


def run_replay(args: argparse.Namespace, replay: dict, actions_path: pathlib.Path, render: bool,
               frames_dir: pathlib.Path | None) -> dict:
    cmd = [str(REPLAY_BIN), "--actions", str(actions_path), "--fixture", replay.get("fixture", "m1_tiny_melee"),
           "--attacker", replay["attacker"], "--defender", replay["defender"], "--speed", str(args.speed)]
    for flag, key in (("--attacker-hero", "attacker_hero"), ("--defender-hero", "defender_hero")):
        if replay.get(key):
            cmd += [flag, replay[key]]
    # A duel (a second checkpoint on the defender) is a both-side recording whatever its stamp
    # says: capture_replay once clobbered the duel stamp to "attacker", and replaying a
    # two-sided action stream one-sided desynchronizes at the first defender decision (#43).
    # Deriving the side from the recording's substance keeps those vendored duels replayable.
    side = "both" if replay.get("defender_checkpoint") else replay.get("side")
    if side:
        cmd += ["--side", side]
    if replay.get("allow_wide", args.allow_wide):
        cmd.append("--allow-wide")
    # Which world seed the battle was fought under. Obstacles derive from it, so replaying a
    # recording made on another variant reaches a different board and the action stream stops
    # matching; the 2026-08-09 loss capture failed verification here for exactly that reason,
    # rejecting 7 of its actions. Recordings without the stamp predate `--battlefield` and were
    # all made on variant 0, which is the default the worker uses when the flag is absent.
    if replay.get("battlefield"):
        cmd += ["--seed-offset", str(replay["battlefield"])]
    if render:
        cmd.append("--render")
    if frames_dir is not None:
        cmd += ["--frames-dir", str(frames_dir)]

    env = dict(os.environ, FHEROES2_DATA=str(REPO_ROOT))
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    for line in out.stdout.splitlines():
        if '"record":"replay_terminal"' in line:
            return json.loads(line)
    raise SystemExit(f"replay tool produced no terminal record (exit {out.returncode}):\n{out.stderr}")


def check(terminal: dict, replay: dict, label: str) -> None:
    problems = []
    if not terminal["exact"]:
        problems.append(f"not exact: used {terminal['actions_used']}/{terminal['actions_recorded']}, "
                        f"rejected {terminal['rejected']}")
    if terminal["termination"] != replay["termination"]:
        problems.append(f"termination {terminal['termination']} != recorded {replay['termination']}")
    if problems:
        raise SystemExit(f"{label} replay diverged from the recording: " + "; ".join(problems))


def assemble(frames_dir: pathlib.Path, out_path: pathlib.Path, scale: int, hold_last: float) -> None:
    entries = []
    for line in (frames_dir / "manifest.tsv").read_text().splitlines():
        name, ms = line.split("\t")
        entries.append((name, int(ms)))
    if len(entries) < 2:
        raise SystemExit("too few captured frames to assemble")

    concat = []
    for (name, ms), (_, next_ms) in zip(entries, entries[1:]):
        concat.append(f"file '{name}'\nduration {max(next_ms - ms, 1) / 1000.0:.3f}")
    concat.append(f"file '{entries[-1][0]}'\nduration {hold_last:.3f}")
    concat.append(f"file '{entries[-1][0]}'")  # concat demuxer honors the last duration only with a trailing repeat
    (frames_dir / "concat.txt").write_text("\n".join(concat) + "\n")

    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(frames_dir / "concat.txt"),
                    "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                   cwd=frames_dir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay_json")
    parser.add_argument("out_mp4")
    parser.add_argument("--speed", type=int, default=10, help="engine battle speed 1..10 for the rendered pass")
    parser.add_argument("--scale", type=int, default=2, help="integer upscale of the 640x480 frames")
    parser.add_argument("--hold-last", type=float, default=2.0, help="seconds to hold the final frame")
    parser.add_argument("--allow-wide", action="store_true",
                        help="for recordings from before capture_replay stored the flag")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    if not REPLAY_BIN.exists():
        raise SystemExit(f"{REPLAY_BIN} not built; run src/agent_replay/build_replay.sh")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found")

    replay = json.loads(pathlib.Path(args.replay_json).read_text())
    out_path = pathlib.Path(args.out_mp4).resolve()

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="render_replay_", dir=out_path.parent))
    try:
        actions_path = workdir / "actions.txt"
        actions_path.write_text("\n".join(str(f["action"]) for f in replay["frames"]) + "\n")

        headless = run_replay(args, replay, actions_path, render=False, frames_dir=None)
        check(headless, replay, "headless")

        frames_dir = workdir / "frames"
        frames_dir.mkdir()
        rendered = run_replay(args, replay, actions_path, render=True, frames_dir=frames_dir)
        check(rendered, replay, "rendered")
        if rendered["state_digest"] != headless["state_digest"]:
            raise SystemExit("rendered terminal digest differs from headless: the interface changed the battle")

        assemble(frames_dir, out_path, args.scale, args.hold_last)
        frames = len((frames_dir / "manifest.tsv").read_text().splitlines())
        print(f"{out_path.name}: {replay['termination']} in {len(replay['frames'])} decisions, "
              f"{frames} frames, digest {rendered['state_digest'][:12]} verified headless==rendered")
    finally:
        if args.keep_frames:
            print(f"frames kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
