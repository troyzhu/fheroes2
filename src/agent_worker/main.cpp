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

// fheroes2_agent_worker, Milestone 1 shape: run the fixed scenario fixtures headlessly with both
// sides driven by the built-in engine AI, repeat each scenario a number of times, and verify that
// every repetition of a scenario reproduces the identical canonical terminal digest.
//
// This translation unit deliberately lives outside src/fheroes2 so that neither build system's
// game-source glob picks up its main() (agent spec, section 6.1). The JSONL protocol replaces
// this command-line interface in Milestone 4; keep stdout machine-readable in the meantime.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <iostream>
#include <optional>
#include "agent_battle_runner.h"
#include "agent_external_controller.h"
#include "agent_capabilities.h"
#include "agent_scenario.h"
#include "agent_trajectory.h"
#include "logging.h"


namespace
{
    // "1:50,2:10" -> Peasant x50, Archer x10. Returns false on anything malformed, because a
    // silently ignored army specification would produce a battle nobody asked for.
    bool parseSideSpec( const std::string & text, fheroes2::agent::SideSpec & side )
    {
        for ( auto & slot : side.slots ) {
            slot = {};
        }

        size_t slot = 0;
        size_t pos = 0;
        while ( pos < text.size() ) {
            const size_t comma = text.find( ',', pos );
            const std::string item = text.substr( pos, ( comma == std::string::npos ) ? std::string::npos : comma - pos );
            const size_t colon = item.find( ':' );
            if ( colon == std::string::npos || slot >= side.slots.size() ) {
                return false;
            }
            try {
                side.slots[slot].monsterId = std::stoi( item.substr( 0, colon ) );
                side.slots[slot].count = static_cast<uint32_t>( std::stoul( item.substr( colon + 1 ) ) );
            }
            catch ( ... ) {
                return false;
            }
            ++slot;
            if ( comma == std::string::npos ) {
                break;
            }
            pos = comma + 1;
        }
        return slot > 0;
    }
}

int main( int argc, char ** argv )
{
    int runs = 10;
    // Protocol mode: one JSON object per line on stdout, an action index per line on stdin.
    bool protocolMode = false;
    std::string controlledSide = "attacker";
    // Army overrides, "monsterId:count,monsterId:count". Empty leaves the fixture's own armies.
    // This is the difficulty control decisions/0005-training-and-reward.md requires: a generator
    // whose effect is measured as a win rate rather than asserted from army sizes.
    std::string attackerSpec;
    std::string defenderSpec;
    // Number of world seeds per fixture. Each seed is a different battle from the same armies.
    int seedCount = 1;
    std::string onlyFixture;
    std::string trajectoryDir;
    std::string capabilityAuditPath;
    bool auditCoverage = false;
    bool quiet = false;

    for ( int i = 1; i < argc; ++i ) {
        const auto next = [&]( const char * name ) -> const char * {
            if ( i + 1 >= argc ) {
                std::fprintf( stderr, "missing value for %s\n", name );
                std::exit( 2 );
            }
            return argv[++i];
        };

        if ( std::strcmp( argv[i], "--runs" ) == 0 ) {
            runs = std::atoi( next( "--runs" ) );
        }
        else if ( std::strcmp( argv[i], "--seeds" ) == 0 ) {
            seedCount = std::atoi( next( "--seeds" ) );
        }
        else if ( std::strcmp( argv[i], "--protocol" ) == 0 ) {
            protocolMode = true;
        }
        else if ( std::strcmp( argv[i], "--side" ) == 0 ) {
            controlledSide = next( "--side" );
        }
        else if ( std::strcmp( argv[i], "--attacker" ) == 0 ) {
            attackerSpec = next( "--attacker" );
        }
        else if ( std::strcmp( argv[i], "--defender" ) == 0 ) {
            defenderSpec = next( "--defender" );
        }
        else if ( std::strcmp( argv[i], "--fixture" ) == 0 ) {
            onlyFixture = next( "--fixture" );
        }
        else if ( std::strcmp( argv[i], "--list" ) == 0 ) {
            for ( const auto & scenario : fheroes2::agent::milestone1Fixtures() ) {
                std::printf( "%s\n", scenario.scenarioId.c_str() );
            }
            return 0;
        }
        else if ( std::strcmp( argv[i], "--trajectory-dir" ) == 0 ) {
            trajectoryDir = next( "--trajectory-dir" );
        }
        else if ( std::strcmp( argv[i], "--capability-audit" ) == 0 ) {
            capabilityAuditPath = next( "--capability-audit" );
        }
        else if ( std::strcmp( argv[i], "--audit-coverage" ) == 0 ) {
            auditCoverage = true;
        }
        else if ( std::strcmp( argv[i], "--quiet" ) == 0 ) {
            quiet = true;
        }
        else {
            std::fprintf( stderr,
                          "usage: fheroes2_agent_worker [--runs N] [--seeds N] [--protocol] [--side attacker|defender|both]\n       [--attacker id:count,...] [--defender id:count,...] [--fixture ID] [--trajectory-dir DIR] [--audit-coverage] [--capability-audit PATH] [--list] "
                          "[--quiet]\n"
                          "unknown argument: %s\n",
                          argv[i] );
            return 2;
        }
    }

    if ( runs < 1 ) {
        std::fprintf( stderr, "--runs must be at least 1\n" );
        return 2;
    }

    if ( seedCount < 1 ) {
        std::fprintf( stderr, "--seeds must be at least 1\n" );
        return 2;
    }

    Logging::InitLog();

    if ( !capabilityAuditPath.empty() ) {
        // Standalone mode: write the machine-generated monster capability audit and exit.
        if ( !fheroes2::agent::writeCapabilityAudit( capabilityAuditPath ) ) {
            std::fprintf( stderr, "cannot write capability audit: %s\n", capabilityAuditPath.c_str() );
            return 2;
        }
        std::printf( "CAPABILITY_AUDIT path=%s monsters=%zu\n", capabilityAuditPath.c_str(), fheroes2::agent::auditAllMonsters().size() );
        return 0;
    }

    std::vector<fheroes2::agent::Scenario> scenarios;
    for ( const auto & scenario : fheroes2::agent::milestone1Fixtures() ) {
        if ( !onlyFixture.empty() && scenario.scenarioId != onlyFixture ) {
            continue;
        }

        // --seeds N replays a fixture's armies under N different world seeds. The map seed and
        // therefore the obstacle layout and the combat seed all derive from it, so each is a
        // genuinely different battle rather than a repeat, while the army matchup is held
        // fixed. That separation of matchup from seed is what
        // agent_play/docs/rl/scenario-distribution.md requires of a generator: difficulty is a
        // property of the matchup, estimated over seeds.
        for ( int s = 0; s < seedCount; ++s ) {
            fheroes2::agent::Scenario variant = scenario;
            if ( !attackerSpec.empty() && !parseSideSpec( attackerSpec, variant.attacker ) ) {
                std::fprintf( stderr, "cannot parse --attacker %s\n", attackerSpec.c_str() );
                return 2;
            }
            if ( !defenderSpec.empty() && !parseSideSpec( defenderSpec, variant.defender ) ) {
                std::fprintf( stderr, "cannot parse --defender %s\n", defenderSpec.c_str() );
                return 2;
            }
            if ( s > 0 ) {
                variant.worldSeed = scenario.worldSeed + static_cast<uint32_t>( s );
                variant.scenarioId = scenario.scenarioId + "-seed" + std::to_string( s );
            }
            scenarios.push_back( variant );
        }
    }
    if ( scenarios.empty() ) {
        std::fprintf( stderr, "no fixture matches '%s' (use --list)\n", onlyFixture.c_str() );
        return 2;
    }

    for ( const auto & scenario : scenarios ) {
        const std::string error = fheroes2::agent::validateScenario( scenario );
        if ( !error.empty() ) {
            std::fprintf( stderr, "invalid scenario: %s\n", error.c_str() );
            return 2;
        }
    }

    std::fprintf( stderr, "[worker] fixtures=%zu runs=%d seeds=%d\n", scenarios.size(), runs, seedCount );

    if ( protocolMode ) {
        // Protocol v1, the smallest thing that lets an external policy drive a battle.
        //
        //   worker -> {"record":"decision","observation":{...},"legal_actions":[...]}
        //   client -> 411
        //   worker -> {"record":"terminal",...}
        //
        // One JSON object per line on stdout and nothing else on that stream, so a client can
        // parse line by line; diagnostics go to stderr (agent spec section 13).
        fheroes2::agent::ControlledSide side = fheroes2::agent::ControlledSide::Attacker;
        if ( controlledSide == "defender" ) {
            side = fheroes2::agent::ControlledSide::Defender;
        }
        else if ( controlledSide == "both" ) {
            side = fheroes2::agent::ControlledSide::Both;
        }
        else if ( controlledSide != "attacker" ) {
            std::fprintf( stderr, "--side must be attacker, defender or both\n" );
            return 2;
        }

        for ( const auto & scenario : scenarios ) {
            auto decide = []( const fheroes2::agent::Observation & observation,
                              const fheroes2::agent::ActionSet & set ) -> std::optional<uint32_t> {
                std::printf( "{\"record\":\"decision\",\"observation\":%s,\"legal_actions\":[", fheroes2::agent::observationToJson( observation ).c_str() );
                for ( size_t i = 0; i < set.candidates.size(); ++i ) {
                    std::printf( "%s%u", ( i == 0 ) ? "" : ",", set.candidates[i].canonicalIndex );
                }
                std::printf( "]}\n" );
                std::fflush( stdout );

                // Blocking read. The engine owns the call stack, so this waits inside
                // Arena::UnitTurn until the client answers. End of input unwinds the episode.
                std::string line;
                if ( !std::getline( std::cin, line ) ) {
                    return std::nullopt;
                }
                try {
                    return static_cast<uint32_t>( std::stoul( line ) );
                }
                catch ( ... ) {
                    // Malformed input is recoverable: report it and let the controller reject
                    // the selection, which skips the turn rather than killing the worker.
                    std::fprintf( stderr, "[worker] unparseable action %s\n", line.c_str() );
                    return static_cast<uint32_t>( fheroes2::agent::actionSpaceSize );
                }
            };

            fheroes2::agent::ExternalDecisionController controller( side, decide );
            fheroes2::agent::EpisodeRecording recording;
            recording.auditTeacherCoverage = true;

            std::printf( "{\"record\":\"episode_start\",\"scenario_id\":\"%s\",\"controlled_side\":\"%s\"}\n", scenario.scenarioId.c_str(),
                         controlledSide.c_str() );
            std::fflush( stdout );

            const fheroes2::agent::EpisodeOutcome outcome = fheroes2::agent::runEpisode( scenario, &recording, &controller );

            std::printf( "{\"record\":\"terminal\",\"scenario_id\":\"%s\",\"termination\":\"%s\",\"rounds\":%d"
                         ",\"attacker\":{\"live_stacks\":%u,\"live_creatures\":%u,\"hit_points\":%u}"
                         ",\"defender\":{\"live_stacks\":%u,\"live_creatures\":%u,\"hit_points\":%u}"
                         ",\"decisions_seen\":%u,\"decisions_answered\":%u,\"rejected\":%u,\"client_closed\":%s"
                         ",\"state_digest\":\"%s\"}\n",
                         scenario.scenarioId.c_str(), fheroes2::agent::terminationName( outcome.termination ), outcome.rounds, outcome.attacker.liveStacks,
                         outcome.attacker.liveCreatures, outcome.attacker.hitPoints, outcome.defender.liveStacks, outcome.defender.liveCreatures,
                         outcome.defender.hitPoints, controller.decisionsSeen(), controller.decisionsAnswered(), controller.rejectedSelections(),
                         controller.isFinished() ? "true" : "false", outcome.stateDigest.c_str() );
            std::fflush( stdout );
        }

        return 0;
    }


    std::map<std::string, std::set<std::string>> digestsPerScenario;
    std::map<std::string, std::set<std::string>> decisionDigestsPerScenario;

    struct CoverageTotals
    {
        size_t decisions = 0;
        size_t resolved = 0;
        size_t matched = 0;
        uint32_t minCandidates = UINT32_MAX;
    };
    std::map<std::string, CoverageTotals> coveragePerScenario;

    for ( const auto & scenario : scenarios ) {
        for ( int run = 0; run < runs; ++run ) {
            // Passive teacher recording is always on: it must never change the outcome (that
            // invariance is exactly what the golden state digests verify), and it yields the
            // decision-stream digest reported below.
            fheroes2::agent::EpisodeRecording recording;
            recording.auditTeacherCoverage = auditCoverage;
            const fheroes2::agent::EpisodeOutcome outcome = fheroes2::agent::runEpisode( scenario, &recording );

            if ( auditCoverage ) {
                CoverageTotals & totals = coveragePerScenario[scenario.scenarioId];
                for ( const fheroes2::agent::DecisionCoverage & cov : recording.coverage ) {
                    ++totals.decisions;
                    if ( cov.teacherResolved ) {
                        ++totals.resolved;
                    }
                    if ( cov.teacherMatched ) {
                        ++totals.matched;
                    }
                    if ( cov.candidateCount < totals.minCandidates ) {
                        totals.minCandidates = cov.candidateCount;
                    }
                }
            }

            digestsPerScenario[scenario.scenarioId].insert( outcome.stateDigest );
            decisionDigestsPerScenario[scenario.scenarioId].insert( recording.decisionDigest );

            if ( !trajectoryDir.empty() ) {
                char runSuffix[16];
                std::snprintf( runSuffix, sizeof( runSuffix ), "-run%02d", run );

                const std::string path = trajectoryDir + "/" + scenario.scenarioId + runSuffix + ".jsonl";
                fheroes2::agent::TrajectoryWriter writer( path );
                if ( !writer.isOpen() ) {
                    std::fprintf( stderr, "cannot open trajectory file: %s\n", path.c_str() );
                    return 2;
                }

                writer.writeHeader( scenario, outcome.mapSeed, outcome.combatSeed );
                for ( size_t d = 0; d < recording.decisions.size(); ++d ) {
                    // Coverage and observations are recorded only under --audit-coverage, and
                    // are parallel to decisions when they are. Passing them turns each record
                    // from a bare action into a complete behaviour-cloning sample.
                    const fheroes2::agent::DecisionCoverage * cov = ( d < recording.coverage.size() ) ? &recording.coverage[d] : nullptr;
                    const fheroes2::agent::Observation * obs = ( d < recording.observations.size() ) ? &recording.observations[d] : nullptr;
                    writer.writeDecision( recording.decisions[d], cov, obs );
                }
                writer.writeTerminal( outcome, recording.decisions.size(), recording.decisionDigest );
            }

            if ( !quiet && run == 0 ) {
                std::printf( "fixture=%s map_seed=%u combat_seed=%u rounds=%d termination=%s a_stacks=%u a_creatures=%u a_hp=%u d_stacks=%u d_creatures=%u "
                             "d_hp=%u decisions=%zu decision_digest=%s digest=%s\n",
                             scenario.scenarioId.c_str(), outcome.mapSeed, outcome.combatSeed, outcome.rounds,
                             fheroes2::agent::terminationName( outcome.termination ), outcome.attacker.liveStacks, outcome.attacker.liveCreatures,
                             outcome.attacker.hitPoints, outcome.defender.liveStacks, outcome.defender.liveCreatures, outcome.defender.hitPoints,
                             recording.decisions.size(), recording.decisionDigest.c_str(), outcome.stateDigest.c_str() );
            }
        }
    }

    bool deterministic = true;
    for ( const auto & scenario : scenarios ) {
        const size_t distinct = digestsPerScenario[scenario.scenarioId].size();
        const size_t distinctDecisions = decisionDigestsPerScenario[scenario.scenarioId].size();
        std::printf( "RESULT fixture=%s runs=%d distinct_digests=%zu distinct_decision_digests=%zu\n", scenario.scenarioId.c_str(), runs, distinct,
                     distinctDecisions );
        if ( distinct != 1 || distinctDecisions != 1 ) {
            deterministic = false;
        }
    }

    bool coverageComplete = true;
    if ( auditCoverage ) {
        for ( const auto & scenario : scenarios ) {
            const CoverageTotals & totals = coveragePerScenario[scenario.scenarioId];
            const double coverage = ( totals.decisions == 0 ) ? 0.0 : ( 100.0 * static_cast<double>( totals.matched ) / static_cast<double>( totals.decisions ) );
            std::printf( "COVERAGE fixture=%s decisions=%zu resolved=%zu matched=%zu coverage=%.1f%% min_candidates=%u\n", scenario.scenarioId.c_str(),
                         totals.decisions, totals.resolved, totals.matched, coverage, ( totals.decisions == 0 ) ? 0 : totals.minCandidates );
            if ( totals.decisions == 0 || totals.matched != totals.decisions ) {
                coverageComplete = false;
            }
        }
    }

    std::printf( "VERDICT fixtures=%zu runs_each=%d deterministic=%s%s\n", scenarios.size(), runs, deterministic ? "yes" : "no",
                 auditCoverage ? ( coverageComplete ? " teacher_coverage=complete" : " teacher_coverage=INCOMPLETE" ) : "" );

    return ( deterministic && ( !auditCoverage || coverageComplete ) ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
