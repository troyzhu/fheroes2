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

#include <cstdint>
#include <string>
#include <vector>

#include "agent_scenario.h"

namespace Battle
{
    class DecisionController;
}

namespace fheroes2::agent
{
    // Why an episode ended (the Milestone 1 subset of agent spec, section 8.5; the external
    // decision and protocol related reasons arrive with later milestones).
    enum class Termination : uint8_t
    {
        Victory,     // the attacker won
        Defeat,      // the defender won
        EngineDraw,  // the engine reported no winner (e.g. mutual elimination)
        RoundLimit,  // truncated by Scenario::maxRounds while the battle was still valid
    };

    const char * terminationName( const Termination termination );

    // Terminal state of one battle unit, read from the Force objects before the arena is
    // destroyed: the input Army objects are not synchronized after a headless battle.
    struct UnitTerminalState
    {
        uint32_t uid{ 0 };
        int monsterId{ 0 };
        uint32_t count{ 0 };
        uint32_t hitPoints{ 0 };
        int32_t headCell{ -1 };
        bool isAttacker{ false };
        bool isValid{ false };
    };

    struct SideSummary
    {
        uint32_t liveStacks{ 0 };
        uint32_t liveCreatures{ 0 };
        uint32_t hitPoints{ 0 };
    };

    struct EpisodeOutcome
    {
        uint32_t effectiveWorldSeed{ 0 };
        uint32_t mapSeed{ 0 };
        uint32_t combatSeed{ 0 };
        int32_t rounds{ 0 };
        uint32_t attackerResult{ 0 };
        uint32_t defenderResult{ 0 };
        Termination termination{ Termination::EngineDraw };
        SideSummary attacker;
        SideSummary defender;
        // Attacker force units first, then defender force units, in engine iteration order.
        std::vector<UnitTerminalState> units;
        // SHA-256 over the canonical terminal state ("agent_terminal_v1"): seeds, rounds,
        // results, termination and every unit record. Excludes anything non-deterministic.
        std::string stateDigest;
    };

    struct EpisodeRecording;

    // Runs one complete headless AI-vs-AI episode for a scenario that has already passed
    // validateScenario(). Reseeds the process-global thread-local random device to make the
    // generated world reproducible -- deliberate for a dedicated worker process, unacceptable
    // inside the interactive game (agent spec, section 7.2, option 1).
    //
    // One live Battle::Arena is allowed per process; this function owns its arena for the whole
    // call, so callers must not hold another one.
    //
    // When `recording` is non-null, every full-fledged built-in-AI decision is captured
    // passively (typed command snapshots plus a decision-stream digest) through the
    // Battle::DecisionController observer; the battle outcome itself is unaffected.
    // `controller`, when given, plays the side it claims and the built-in AI plays the rest.
    // Passing one alongside a recording still records every decision, including the ones the
    // controller made, so an externally driven episode is auditable the same way a teacher
    // episode is.
    EpisodeOutcome runEpisode( const Scenario & scenario, EpisodeRecording * recording = nullptr, Battle::DecisionController * controller = nullptr );
}
