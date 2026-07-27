#!/bin/bash
# Build the Phase 0 headless battle smoke spike.
#
# Strategy: the plain-Makefile build (`make -C src/dist`) already produces one object file per
# game translation unit. Rather than configuring a second build system, this script compiles only
# smoke_battle.cpp and relinks it against those existing objects, excluding fheroes2.o -- the
# translation unit that holds the game's real main(). That is the spec's Section 6.4 fallback
# ("compile the non-entry sources a second time") reduced to a relink, and it directly exercises
# the constraint that the agent entry point must not collide with the game entry point.
#
# Prerequisite: `make -C src/dist -j10` has been run at least once.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPIKE_DIR="${REPO_ROOT}/agent_play/spike"
OBJ_DIR="${REPO_ROOT}/src/dist/fheroes2"
ENGINE_LIB="${REPO_ROOT}/src/dist/engine/libengine.a"
SMACKER_LIB="${REPO_ROOT}/src/dist/thirdparty/libsmacker/libsmacker.a"
OUT="${SPIKE_DIR}/smoke_battle"

if [ ! -f "${OBJ_DIR}/fheroes2.o" ]; then
    echo "error: ${OBJ_DIR}/fheroes2.o not found." >&2
    echo "       Run 'make -C src/dist -j10' from the repo root first." >&2
    exit 1
fi

# Flags are kept in arrays so that a repo path containing spaces (e.g. an external
# drive mounted as "/Volumes/External Drive") survives expansion intact.
SRC="${REPO_ROOT}/src"
INCLUDES=()
for d in agg ai army audio battle campaign castle dialog editor game gui h2d heroes image kingdom maps monster resource spell system world; do
    INCLUDES+=( "-I${SRC}/fheroes2/${d}" )
done
INCLUDES+=( "-I${SRC}/engine" "-I${SRC}/thirdparty/libsmacker" )

# sdl2-config output may hold several flags, so word-splitting it is intentional;
# Homebrew/system prefixes are assumed space-free.
if command -v sdl2-config >/dev/null 2>&1; then
    read -ra SDL_CFLAGS <<< "$(sdl2-config --cflags)"
    read -ra SDL_LIBS <<< "$(sdl2-config --libs)"
else
    SDL_CFLAGS=( -I/opt/homebrew/include/SDL2 -D_THREAD_SAFE )
    SDL_LIBS=( -L/opt/homebrew/lib -lSDL2 )
fi

echo "[1/2] compiling smoke_battle.cpp"
c++ -c -o "${SPIKE_DIR}/smoke_battle.o" "${SPIKE_DIR}/smoke_battle.cpp" \
    "${INCLUDES[@]}" "${SDL_CFLAGS[@]}" \
    -fsigned-char -pthread -O2 -std=c++17 -Wall -Wextra

# Every game object except the one carrying the real main().
GAME_OBJS=()
for o in "${OBJ_DIR}"/*.o; do
    [ "$(basename "$o")" = "fheroes2.o" ] && continue
    GAME_OBJS+=("$o")
done

echo "[2/2] linking ${#GAME_OBJS[@]} game objects (fheroes2.o excluded) -> $(basename "${OUT}")"
c++ -o "${OUT}" \
    "${SPIKE_DIR}/smoke_battle.o" \
    "${GAME_OBJS[@]}" \
    "${ENGINE_LIB}" "${SMACKER_LIB}" \
    -lSDL2_mixer "${SDL_LIBS[@]}" -lz -pthread

echo "built: ${OUT}"
