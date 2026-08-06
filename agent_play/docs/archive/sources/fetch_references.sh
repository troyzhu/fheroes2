#!/bin/bash
# Fetch local copies of every source cited by the two verified research runs
# (the two research runs archived under agent_play/docs/archive/research-runs/).
#
# Reproducible: re-running overwrites files/ and regenerates manifest.tsv (url, status, bytes,
# scraped title where available). Files larger than MAX_BYTES are skipped and marked in the
# manifest. Paywalled pages (e.g. Nature) yield whatever the anonymous fetch returns — the
# manifest records the byte count so a stub is visible.

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES="${DIR}/files"
MANIFEST="${DIR}/manifest.tsv"
MAX_BYTES=$((25 * 1024 * 1024))
UA="Mozilla/5.0 (references-fetcher; fheroes2 agent docs)"

mkdir -p "${FILES}"
printf 'file\tstatus\tbytes\ttitle_or_note\turl\n' > "${MANIFEST}"

note() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "${MANIFEST}"; }

fetch() { # fetch <outfile> <url>
    local out="$1" url="$2"
    if curl -sSfL --max-time 120 -A "${UA}" -o "${FILES}/${out}" "${url}"; then
        local bytes
        bytes=$(stat -f%z "${FILES}/${out}" 2>/dev/null || stat -c%s "${FILES}/${out}")
        if [ "${bytes}" -gt "${MAX_BYTES}" ]; then
            rm -f "${FILES}/${out}"
            note "${out}" "SKIPPED_TOO_LARGE" "${bytes}" "-" "${url}"
        else
            note "${out}" "OK" "${bytes}" "-" "${url}"
        fi
    else
        note "${out}" "FAILED" "0" "-" "${url}"
    fi
    sleep 1
}

arxiv() { # arxiv <id>  -> PDF + scraped title
    local id="$1"
    local title
    title=$(curl -sSfL --max-time 60 -A "${UA}" "https://arxiv.org/abs/${id}" 2>/dev/null \
        | sed -n 's/.*<title>\[[^]]*\] *\(.*\)<\/title>.*/\1/p' | head -1)
    if curl -sSfL --max-time 120 -A "${UA}" -o "${FILES}/arxiv-${id}.pdf" "https://arxiv.org/pdf/${id}"; then
        local bytes
        bytes=$(stat -f%z "${FILES}/arxiv-${id}.pdf" 2>/dev/null || stat -c%s "${FILES}/arxiv-${id}.pdf")
        if [ "${bytes}" -gt "${MAX_BYTES}" ]; then
            rm -f "${FILES}/arxiv-${id}.pdf"
            note "arxiv-${id}.pdf" "SKIPPED_TOO_LARGE" "${bytes}" "${title:-?}" "https://arxiv.org/abs/${id}"
        else
            note "arxiv-${id}.pdf" "OK" "${bytes}" "${title:-?}" "https://arxiv.org/abs/${id}"
        fi
    else
        note "arxiv-${id}.pdf" "FAILED" "0" "${title:-?}" "https://arxiv.org/abs/${id}"
    fi
    sleep 1
}

# ---- arXiv papers (both runs) ----------------------------------------------
for id in 1708.04782 2105.13807 2006.14171 2011.06363 2006.13760 1912.06680 2206.02855 \
          2105.11674 1710.06542 2110.05038 2308.03526 2104.06890 2009.05643 1811.06447 \
          2410.17647 2104.03113 2305.19240 2607.06514 \
          2602.04879 2607.00152 2503.20783 2006.05990 2504.04395 2012.01914; do
    arxiv "${id}"
done

# ---- arXiv papers (2026-08-05 evening review: anchoring, value, search) ----
for id in 1712.01815 2106.08909 1812.02900 1509.06461; do
    arxiv "${id}"
done

# UCT (Kocsis & Szepesvari, ECML 2006) predates arXiv posting; Stanford GGP mirror.
fetch "uct-kocsis-szepesvari-2006.pdf"  "http://ggp.stanford.edu/readings/uct.pdf"

# ---- GitHub raw files (small, canonical content) ---------------------------
fetch "pysc2-environment.md"            "https://raw.githubusercontent.com/google-deepmind/pysc2/master/docs/environment.md"
fetch "pysc2-README.md"                 "https://raw.githubusercontent.com/google-deepmind/pysc2/master/README.md"
fetch "alphastar-detailed-architecture.txt" "https://raw.githubusercontent.com/chengyu2/learning_alpha_star/master/detailed-architecture.txt"
fetch "vcmi-gym-README.md"              "https://raw.githubusercontent.com/smanolloff/vcmi-gym/main/README.md"
fetch "microrts-py-README.md"           "https://raw.githubusercontent.com/Farama-Foundation/MicroRTS-Py/master/README.md"
fetch "entity-gym-README.md"            "https://raw.githubusercontent.com/entity-neural-network/entity-gym/main/README.md"
fetch "rogue-net-README.md"             "https://raw.githubusercontent.com/entity-neural-network/rogue-net/main/README.md"
fetch "nle-README.md"                   "https://raw.githubusercontent.com/heiner/nle/main/README.md"
fetch "griddly-README.md"               "https://raw.githubusercontent.com/Bam4d/Griddly/develop/README.md"
fetch "arlinbfw-README.md"              "https://raw.githubusercontent.com/DStelter94/ARLinBfW/master/README.md"
fetch "sample-factory-README.md"        "https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/README.md"
fetch "lux-ai-2021-winner-README.md"    "https://raw.githubusercontent.com/IsaiahPressman/Kaggle_Lux_AI_2021/main/README.md"
fetch "mmai-README.md"                  "https://raw.githubusercontent.com/vcmi-mods/mmai/vcmi-1.7/README.md"
fetch "invalid-action-masking-README.md" "https://raw.githubusercontent.com/vwxyzjn/invalid-action-masking/master/README.md"
fetch "metamon-README.md"               "https://raw.githubusercontent.com/UT-Austin-RPL/metamon/main/README.md"
fetch "poke-env-README.md"              "https://raw.githubusercontent.com/hsahovic/poke-env/master/README.md"

# ---- Web pages (HTML snapshots) --------------------------------------------
fetch "vcmi-gym-blog.html"              "https://smanolloff.github.io/projects/vcmi-gym/"
fetch "stratega-page.html"              "https://gaigresearch.github.io/2020/06/15/dockhorn2020stratega/"
fetch "sb3-maskable-ppo.html"           "https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html"
fetch "ppo-implementation-details.html" "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"
fetch "entity-based-rl-blog.html"       "https://clemenswinter.com/2023/04/14/entity-based-reinforcement-learning/"
fetch "alphastar-deepmind-blog.html"    "https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/"
fetch "alphastar-nature-landing.html"   "https://www.nature.com/articles/s41586-019-1724-z"
fetch "griddly-observation-spaces.html" "https://griddly.readthedocs.io/en/latest/getting-started/observation%20spaces/index.html"
fetch "griddly-rllib-intro.html"        "https://griddly.readthedocs.io/en/latest/rllib/intro/index.html"
fetch "minihack-observation-spaces.html" "https://minihack.readthedocs.io/en/latest/getting-started/observation_spaces.html"
fetch "nle-revisited-blog.html"         "https://iclr-blogposts.github.io/2026/blog/2026/revisiting-the-nle/"
fetch "alphastar-unformatted.pdf"       "https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf"

echo "done: $(grep -c $'\tOK\t' "${MANIFEST}") fetched, $(grep -c $'\tFAILED\t' "${MANIFEST}") failed, $(grep -c "SKIPPED" "${MANIFEST}") skipped"
echo "manifest: ${MANIFEST}"
