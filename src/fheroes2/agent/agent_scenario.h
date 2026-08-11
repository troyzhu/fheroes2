/***************************************************************************
 *   fheroes2: https://github.com/ihhub/fheroes2                           *
 *   Copyright (C) 2026                                                    *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fheroes2::agent
{
    // One slot per army position. Slot order is part of the engine-compatible combat seed and of
    // initial battlefield placement, so positions are always represented explicitly (agent spec,
    // section 11.2).
    struct StackSpec
    {
        // Monster id as defined by the Monster class; 0 is Monster::UNKNOWN and marks an empty slot.
        int monsterId{ 0 };
        uint32_t count{ 0 };

        bool isEmpty() const
        {
            return count == 0;
        }
    };

    // Checked against Army::maximumTroopCount where the Army headers are available.
    inline constexpr size_t scenarioSlotCount{ 5 };

    // Safety cap from the scenario validation rules (agent spec, section 11.1).
    inline constexpr uint32_t scenarioMaxStackCount{ 500000 };

    struct SideSpec
    {
        std::array<StackSpec, scenarioSlotCount> slots{};
    };

    // Optional per-side hero commander. Real maps always have one, and every battle unit's
    // effective attack and defense include the commander's stats, so a commander-less scenario
    // understates both sides. Absent by default, which keeps every existing scenario and its
    // digests bit-identical.
    struct CommanderSpec
    {
        bool present{ false };
        int attack{ 0 };
        int defense{ 0 };
    };

    // Upper bound for a commander stat. The strongest map hero seen so far carries attack 30;
    // the cap leaves headroom without admitting nonsense.
    inline constexpr int scenarioMaxCommanderStat{ 99 };

    // A fixed creature-only field battle on the 2 x 2 Battle Only world: the C++ counterpart of
    // scenario schema v1 (agent spec, section 11) restricted to what Milestone 1 needs. JSON
    // parsing arrives with the protocol worker; until then scenarios are constructed in code.
    struct Scenario
    {
        std::string scenarioId;
        // A Maps::Ground ground type value. UNKNOWN (0) is rejected by validation.
        int groundType{ 0 };
        // Fixed by the battle profile; anything except 1 is rejected (agent spec, section 7.5).
        int32_t tileIndex{ 1 };
        uint32_t worldSeed{ 0 };
        // Added 2026-08-10 to make one ablation expressible and nothing else. The battle's random
        // stream is seeded from computeBattleSeed( tileIndex, mapSeed, armies ), all of which the
        // world seed fixes, so a search side environment pinned to the live world seed inherits the
        // live battle's exact combat rolls. That is indistinguishable from a perfect model unless
        // the two can be separated, and separating them is the only way to tell whether root search
        // is planning or reading the future. Nonzero perturbs the combat stream while leaving the
        // map, the obstacle layout and the armies identical. Zero is the shipped behaviour and every
        // existing transcript and digest is bit-identical under it.
        uint32_t combatSeedOffset{ 0 };
        int32_t maxRounds{ 100 };
        // ADR 0004's planes_v1 obstacle layer on every serialized observation, off by default
        // so all existing transcripts and digests stay byte-identical.
        bool observeObstacles{ false };
        SideSpec attacker;
        SideSpec defender;
        CommanderSpec attackerCommander;
        CommanderSpec defenderCommander;
        // Admits wide (two-cell) walkers on either side, the wide_v1 profile. Off by default,
        // which keeps every existing scenario and its golden digests bit-identical.
        bool allowWideUnits{ false };
        // flying_v1, off by default so every existing scenario and its golden digests are unchanged.
        bool allowFlyingUnits{ false };
    };

    // Returns an empty string when the scenario is valid, otherwise a human-readable reason for
    // the first violated rule.
    std::string validateScenario( const Scenario & scenario );

    // The fixed Milestone 1 fixture suite: tiny one-stack, three-stack, five-stack, ranged-heavy
    // and a longer balanced battle (the workload shapes of agent spec, section 19.2). All
    // fixtures pass validateScenario().
    const std::vector<Scenario> & milestone1Fixtures();
}
