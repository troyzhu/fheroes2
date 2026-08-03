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
#include "agent_observation.h"

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

    // Per-decision simple_v1 coverage data (Milestone 3): how many legal candidates existed
    // and whether the teacher's chosen action mapped onto the canonical action space.
    struct DecisionCoverage
    {
        uint32_t candidateCount{ 0 };
        bool teacherResolved{ false };
        bool teacherMatched{ false };
        uint32_t teacherCanonicalIndex{ 0 };
        // The legal set as canonical indices, ascending. Stored as a list rather than as the
        // 793-wide mask because 5 to 30 entries are legal at a typical decision, and the two
        // are interconvertible. This plus the observation and the teacher index is one
        // behaviour-cloning sample.
        std::vector<uint32_t> legalActions;
    };

    // Filled by runEpisode() when a recording is requested.
    struct EpisodeRecording
    {
        // Input: when true, runEpisode() additionally enumerates the simple_v1 action set at
        // every full-fledged decision and records teacher coverage against it.
        bool auditTeacherCoverage{ false };

        std::vector<DecisionRecord> decisions;
        // Parallel to `decisions` when auditTeacherCoverage was set.
        std::vector<DecisionCoverage> coverage;
        // Parallel to `decisions` when auditTeacherCoverage was set: the board as a policy
        // would have seen it, captured before the teacher's commands are applied.
        std::vector<Observation> observations;
        // SHA-256 over the canonical byte serialization of every decision
        // ("agent_decisions_v0"): replay-equality of the chosen-command stream.
        std::string decisionDigest;
    };

    std::string computeDecisionDigest( const std::vector<DecisionRecord> & decisions );

    // Writes one episode as JSON Lines: an episode_header record, one decision record per
    // full-fledged decision, and a terminal record.
    //
    // Schema tag "agent_passive_v1": still a deliberate subset of the spec section 15 v1
    // records — no observations, legal-action lists or teacher-matching yet (those arrive with
    // Milestone 4), and no wall-clock fields, so identical episodes produce byte-identical
    // files (determinism first; timestamps can be reintroduced as explicitly-excluded metadata
    // later).
    //
    // v1 adds per-unit terminal state and the two side summaries to the terminal record. Those
    // were previously computed, folded into the state digest and discarded, so the extracted
    // state was verifiable only as a hash and could not be read by a human at all. Serializing
    // them is what makes the interface comparison in the roadmap possible.
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
        // `coverage` and `observation` are optional; passing them writes a complete
        // behaviour-cloning sample rather than an action alone.
        void writeDecision( const DecisionRecord & decision, const DecisionCoverage * coverage = nullptr, const Observation * observation = nullptr );
        void writeTerminal( const EpisodeOutcome & outcome, const size_t decisionCount, const std::string & decisionDigest );

    private:
        std::ofstream _out;
    };
}
