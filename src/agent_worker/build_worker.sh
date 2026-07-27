#!/bin/bash
# Build fheroes2_agent_worker against the Makefile-built game objects.
#
# Same relink strategy as agent_play/spike/build_spike.sh: the worker's main.cpp is the only new
# translation unit; everything else (including src/fheroes2/agent/*) is already compiled into
# src/dist/fheroes2/*.o by `make -C src/dist`. fheroes2.o (the game's main) is excluded.
# Honors FHEROES2_WITH_ASAN / FHEROES2_WITH_TSAN like the engine Makefile.
#
# The CMake target for the worker (option ENABLE_AGENT) is planned for Milestone 4.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER_DIR="${REPO_ROOT}/src/agent_worker"
OBJ_DIR="${REPO_ROOT}/src/dist/fheroes2"
ENGINE_LIB="${REPO_ROOT}/src/dist/engine/libengine.a"
SMACKER_LIB="${REPO_ROOT}/src/dist/thirdparty/libsmacker/libsmacker.a"
OUT="${WORKER_DIR}/fheroes2_agent_worker"

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

# bash 3.2 (macOS) errors on "${arr[@]}" for an empty array under `set -u`.
SAN_FLAGS=()
if [ -n "${FHEROES2_WITH_ASAN:-}" ] || [ -n "${FHEROES2_WITH_TSAN:-}" ]; then
    SANITIZERS="undefined"
    [ -n "${FHEROES2_WITH_ASAN:-}" ] && SANITIZERS="${SANITIZERS},address"
    [ -n "${FHEROES2_WITH_TSAN:-}" ] && SANITIZERS="${SANITIZERS},thread"
    SAN_FLAGS=( "-fsanitize=${SANITIZERS}" )
fi

echo "[1/2] compiling agent_worker main.cpp"
c++ -c -o "${WORKER_DIR}/main.o" "${WORKER_DIR}/main.cpp" \
    "${INCLUDES[@]}" "${SDL_CFLAGS[@]}" ${SAN_FLAGS[@]+"${SAN_FLAGS[@]}"} \
    -fsigned-char -pthread -O2 -std=c++17 -Wall -Wextra

GAME_OBJS=()
for o in "${OBJ_DIR}"/*.o; do
    [ "$(basename "$o")" = "fheroes2.o" ] && continue
    GAME_OBJS+=( "$o" )
done

echo "[2/2] linking ${#GAME_OBJS[@]} game objects (fheroes2.o excluded) -> $(basename "${OUT}")"
c++ -o "${OUT}" \
    "${WORKER_DIR}/main.o" \
    "${GAME_OBJS[@]}" \
    "${ENGINE_LIB}" "${SMACKER_LIB}" \
    ${SAN_FLAGS[@]+"${SAN_FLAGS[@]}"} -lSDL2_mixer "${SDL_LIBS[@]}" -lz -pthread

echo "built: ${OUT}"
