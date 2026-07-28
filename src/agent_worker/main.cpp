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

#include "agent_battle_runner.h"
#include "agent_scenario.h"
#include "agent_trajectory.h"
#include "logging.h"

int main( int argc, char ** argv )
{
    int runs = 10;
    std::string onlyFixture;
    std::string trajectoryDir;
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
        else if ( std::strcmp( argv[i], "--quiet" ) == 0 ) {
            quiet = true;
        }
        else {
            std::fprintf( stderr,
                          "usage: fheroes2_agent_worker [--runs N] [--fixture ID] [--trajectory-dir DIR] [--list] [--quiet]\n"
                          "unknown argument: %s\n",
                          argv[i] );
            return 2;
        }
    }

    if ( runs < 1 ) {
        std::fprintf( stderr, "--runs must be at least 1\n" );
        return 2;
    }

    Logging::InitLog();

    std::vector<fheroes2::agent::Scenario> scenarios;
    for ( const auto & scenario : fheroes2::agent::milestone1Fixtures() ) {
        if ( onlyFixture.empty() || scenario.scenarioId == onlyFixture ) {
            scenarios.push_back( scenario );
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

    std::fprintf( stderr, "[worker] fixtures=%zu runs=%d\n", scenarios.size(), runs );

    std::map<std::string, std::set<std::string>> digestsPerScenario;
    std::map<std::string, std::set<std::string>> decisionDigestsPerScenario;

    for ( const auto & scenario : scenarios ) {
        for ( int run = 0; run < runs; ++run ) {
            // Passive teacher recording is always on: it must never change the outcome (that
            // invariance is exactly what the golden state digests verify), and it yields the
            // decision-stream digest reported below.
            fheroes2::agent::EpisodeRecording recording;
            const fheroes2::agent::EpisodeOutcome outcome = fheroes2::agent::runEpisode( scenario, &recording );

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
                for ( const fheroes2::agent::DecisionRecord & decision : recording.decisions ) {
                    writer.writeDecision( decision );
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

    std::printf( "VERDICT fixtures=%zu runs_each=%d deterministic=%s\n", scenarios.size(), runs, deterministic ? "yes" : "no" );

    return deterministic ? EXIT_SUCCESS : EXIT_FAILURE;
}
