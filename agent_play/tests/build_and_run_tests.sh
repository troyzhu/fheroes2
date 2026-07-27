#!/bin/bash
# Build and run every agent C++ test (agent_play/tests/test_*.cpp).
#
# Same strategy as build_spike.sh: each test TU is compiled on its own and relinked against the
# already-built game objects minus fheroes2.o (the game's main). Honors FHEROES2_WITH_ASAN /
# FHEROES2_WITH_TSAN the same way the engine Makefile does.
#
# Prerequisite: `make -C src/dist -j…` has been run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TESTS_DIR="${REPO_ROOT}/agent_play/tests"
OBJ_DIR="${REPO_ROOT}/src/dist/fheroes2"
ENGINE_LIB="${REPO_ROOT}/src/dist/engine/libengine.a"
SMACKER_LIB="${REPO_ROOT}/src/dist/thirdparty/libsmacker/libsmacker.a"

if [ ! -f "${OBJ_DIR}/fheroes2.o" ]; then
    echo "error: game objects not built. Run 'make -C src/dist -j…' first." >&2
    exit 1
fi

SRC="${REPO_ROOT}/src"
INCLUDES=()
for d in agg ai army audio battle campaign castle dialog editor game gui h2d heroes image kingdom maps monster resource spell system world; do
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

GAME_OBJS=()
for o in "${OBJ_DIR}"/*.o; do
    [ "$(basename "$o")" = "fheroes2.o" ] && continue
    GAME_OBJS+=( "$o" )
done

overall=0
for src in "${TESTS_DIR}"/test_*.cpp; do
    name="$(basename "${src}" .cpp)"
    bin="${TESTS_DIR}/${name}"

    echo "== ${name}: build"
    c++ -c -o "${bin}.o" "${src}" \
        "${INCLUDES[@]}" "${SDL_CFLAGS[@]}" ${SAN_FLAGS[@]+"${SAN_FLAGS[@]}"} \
        -fsigned-char -pthread -O2 -std=c++17 -Wall -Wextra
    c++ -o "${bin}" "${bin}.o" "${GAME_OBJS[@]}" "${ENGINE_LIB}" "${SMACKER_LIB}" \
        ${SAN_FLAGS[@]+"${SAN_FLAGS[@]}"} -lSDL2_mixer "${SDL_LIBS[@]}" -lz -pthread

    echo "== ${name}: run"
    if ! "${bin}"; then
        overall=1
    fi
done

if [ "${overall}" -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "TEST FAILURES PRESENT" >&2
fi
exit "${overall}"
