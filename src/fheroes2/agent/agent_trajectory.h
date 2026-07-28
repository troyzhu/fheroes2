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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "agent_command_snapshot.h"

namespace fheroes2::agent
{
    struct EpisodeOutcome;
    struct Scenario;

    // One full-fledged engine decision, captured passively through
    // Battle::DecisionController::observeChosenActions (agent spec, sections 9.4 and 15.2).
    struct DecisionRecord
    {
        uint32_t engineDecisionIndex{ 0 };
        uint32_t unitUid{ 0 };
        std::vector<CommandSnapshot> actions;
    };

    // Filled by runEpisode() when a recording is requested.
    struct EpisodeRecording
    {
        std::vector<DecisionRecord> decisions;
        // SHA-256 over the canonical byte serialization of every decision
        // ("agent_decisions_v0"): replay-equality of the chosen-command stream.
        std::string decisionDigest;
    };

    std::string computeDecisionDigest( const std::vector<DecisionRecord> & decisions );

    // Writes one episode as JSON Lines: an episode_header record, one decision record per
    // full-fledged decision, and a terminal record.
    //
    // Schema tag "agent_passive_v0": a deliberate subset of the spec section 15 v1 records —
    // no observations, legal-action lists or teacher-matching yet (those arrive with
    // Milestones 3-4), and no wall-clock fields, so identical episodes produce byte-identical
    // files (determinism first; timestamps can be reintroduced as explicitly-excluded metadata
    // later).
    class TrajectoryWriter
    {
    public:
        explicit TrajectoryWriter( const std::string & filePath );
        TrajectoryWriter( const TrajectoryWriter & ) = delete;
        TrajectoryWriter & operator=( const TrajectoryWriter & ) = delete;
        ~TrajectoryWriter() = default;

        bool isOpen() const
        {
            return _out.is_open();
        }

        void writeHeader( const Scenario & scenario, const uint32_t mapSeed, const uint32_t combatSeed );
        void writeDecision( const DecisionRecord & decision );
        void writeTerminal( const EpisodeOutcome & outcome, const size_t decisionCount, const std::string & decisionDigest );

    private:
        std::ofstream _out;
    };
}
