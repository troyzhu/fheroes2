#!/bin/bash
# Build fheroes2_agent_replay against the Makefile-built game objects.
#
# Same relink strategy as src/agent_worker/build_worker.sh: main.cpp here is the only new
# translation unit; everything else (including src/fheroes2/agent/*) is already compiled into
# src/dist/fheroes2/*.o by `make -C src/dist`. fheroes2.o (the game's main) is excluded.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPLAY_DIR="${REPO_ROOT}/src/agent_replay"
OBJ_DIR="${REPO_ROOT}/src/dist/fheroes2"
ENGINE_LIB="${REPO_ROOT}/src/dist/engine/libengine.a"
SMACKER_LIB="${REPO_ROOT}/src/dist/thirdparty/libsmacker/libsmacker.a"
OUT="${REPLAY_DIR}/fheroes2_agent_replay"

if [ ! -f "${OBJ_DIR}/fheroes2.o" ]; then
    echo "error: game objects not built. Run 'make -C src/dist -j…' first." >&2
    exit 1
fi

SRC="${REPO_ROOT}/src"
INCLUDES=()
for d in agent agg ai army audio battle campaign castle dialog editor game gui h2d heroes image kingdom maps monster resource spell system world; do
    INCLUDES+=( "-I${SRC}/fheroes2/${d}" )
done
INCLUDES+=( "-I${SRC}/engine" "-I${SRC}/thirdparty/libsmacker" )

if command -v sdl2-config >/dev/null 2>&1; then
    read -ra SDL_CFLAGS <<< "$(sdl2-config --cflags)"
    read -ra SDL_LIBS <<< "$(sdl2-config --libs)"
else
    SDL_CFLAGS=( -I/opt/homebrew/include/SDL2 -D_THREAD_SAFE )
    SDL_LIBS=( -L/opt/homebrew/lib -lSDL2 )
fi

echo "[1/2] compiling agent_replay main.cpp"
c++ -c -o "${REPLAY_DIR}/main.o" "${REPLAY_DIR}/main.cpp" \
    "${INCLUDES[@]}" "${SDL_CFLAGS[@]}" \
    -fsigned-char -pthread -O2 -std=c++17 -Wall -Wextra

GAME_OBJS=()
for o in "${OBJ_DIR}"/*.o; do
    [ "$(basename "$o")" = "fheroes2.o" ] && continue
    GAME_OBJS+=( "$o" )
done

echo "[2/2] linking ${#GAME_OBJS[@]} game objects (fheroes2.o excluded) -> $(basename "${OUT}")"
c++ -o "${OUT}" \
    "${REPLAY_DIR}/main.o" \
    "${GAME_OBJS[@]}" \
    "${ENGINE_LIB}" "${SMACKER_LIB}" \
    -lSDL2_mixer "${SDL_LIBS[@]}" -lz -pthread

echo "built: ${OUT}"
