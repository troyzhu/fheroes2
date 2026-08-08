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
#include "heroes.h"
#include "maps_tiles_helper.h"
#include "army_troop.h"
#include "maps_fileinfo.h"
#include "world.h"
#include "settings.h"
#include "army.h"
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

    // "13:12" -> commander with attack 13 and defense 12. Range checking stays in
    // validateScenario, which is the single place scenario rules live.
    bool parseHeroSpec( const std::string & text, fheroes2::agent::CommanderSpec & commander )
    {
        const size_t colon = text.find( ':' );
        if ( colon == std::string::npos ) {
            return false;
        }
        try {
            commander.attack = std::stoi( text.substr( 0, colon ) );
            commander.defense = std::stoi( text.substr( colon + 1 ) );
        }
        catch ( ... ) {
            return false;
        }
        commander.present = true;
        return true;
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
    // Hero commander stats, "attack:defense". Empty means no commander, which is the historical
    // behaviour and what every golden digest was recorded under. Real maps always have one, so a
    // faithful reproduction of a map fight needs these.
    std::string attackerHeroSpec;
    std::string defenderHeroSpec;
    // Admits wide (two-cell) walkers, the wide_v1 profile. Off by default.
    bool allowWideUnits = false;
    // Dump a map's starting heroes and neutral stacks, so a real scenario can be reproduced as a
    // fixture instead of guessed at. Uses the engine's own loader, because monster counts are
    // computed during load rather than stored verbatim.
    std::string dumpMapPath;
    // Number of world seeds per fixture. Each seed is a different battle from the same armies.
    int seedCount = 1;
    // First seed index of the run, so two processes can agree on which battlefield variant they
    // are playing: a search side-environment replays the live environment's battlefield only if
    // both start their cycle at the same offset.
    int seedOffset = 0;
    std::string onlyFixture;
    std::string trajectoryDir;
    std::string capabilityAuditPath;
    bool auditCoverage = false;
    bool quiet = false;
    // Protocol mode only: ask the built-in planner for its own choice at every controlled
    // decision and emit it as "teacher_action", the DAgger relabeling query.
    bool probeTeacher = false;
    bool observePlanes = false;

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
        else if ( std::strcmp( argv[i], "--seed-offset" ) == 0 ) {
            seedOffset = std::atoi( next( "--seed-offset" ) );
        }
        else if ( std::strcmp( argv[i], "--protocol" ) == 0 ) {
            protocolMode = true;
        }
        else if ( std::strcmp( argv[i], "--probe-teacher" ) == 0 ) {
            probeTeacher = true;
        }
        else if ( std::strcmp( argv[i], "--planes" ) == 0 ) {
            // ADR 0004's planes_v1: the obstacle layer joins every serialized observation, in
            // protocol decisions and in recorded trajectories alike.
            observePlanes = true;
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
        else if ( std::strcmp( argv[i], "--attacker-hero" ) == 0 ) {
            attackerHeroSpec = next( "--attacker-hero" );
        }
        else if ( std::strcmp( argv[i], "--defender-hero" ) == 0 ) {
            defenderHeroSpec = next( "--defender-hero" );
        }
        else if ( std::strcmp( argv[i], "--allow-wide" ) == 0 ) {
            allowWideUnits = true;
        }
        else if ( std::strcmp( argv[i], "--dump-map" ) == 0 ) {
            dumpMapPath = next( "--dump-map" );
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
                          "usage: fheroes2_agent_worker [--runs N] [--seeds N] [--protocol] [--probe-teacher] [--planes] [--side attacker|defender|both]\n       [--attacker id:count,...] [--defender id:count,...]\n       [--attacker-hero atk:def] [--defender-hero atk:def] [--allow-wide] [--fixture ID] [--trajectory-dir DIR] [--audit-coverage] [--capability-audit PATH] [--list] "
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

    if ( !dumpMapPath.empty() ) {
        // Players and races come from the map's own header. Initializing them by hand trips an
        // assertion in race handling, because a colour with no race set is not a valid player.
        Maps::FileInfo mapInfo;
        if ( !mapInfo.readMP2Map( dumpMapPath, false ) ) {
            std::fprintf( stderr, "cannot read map header %s\n", dumpMapPath.c_str() );
            return 2;
        }

        Settings & conf = Settings::Get();
        conf.setCurrentMapInfo( mapInfo );
        conf.GetPlayers().Init( mapInfo );
        conf.GetPlayers().SetStartGame();

        const bool isOriginal = ( mapInfo.version == GameVersion::SUCCESSION_WARS );
        if ( !world.LoadMapMP2( dumpMapPath, isOriginal ) ) {
            std::fprintf( stderr, "cannot load map %s\n", dumpMapPath.c_str() );
            return 2;
        }

        std::printf( "{\"record\":\"map\",\"path\":\"%s\",\"width\":%d,\"height\":%d}\n", dumpMapPath.c_str(), world.w(), world.h() );

        const auto dumpArmy = []( const Army & army ) {
            bool first = true;
            for ( size_t s = 0; s < army.Size(); ++s ) {
                const Troop * troop = army.GetTroop( s );
                if ( troop == nullptr || !troop->isValid() ) {
                    continue;
                }
                std::printf( "%s{\"monster_id\":%d,\"name\":\"%s\",\"count\":%u}", first ? "" : ",", troop->GetID(), troop->GetName(), troop->GetCount() );
                first = false;
            }
        };

        for ( int32_t index = 0; index < world.w() * world.h(); ++index ) {
            const Maps::Tile & tile = world.getTile( index );
            const MP2::MapObjectType type = tile.getMainObjectType();
            const int32_t x = index % world.w();
            const int32_t y = index / world.w();

            if ( type == MP2::OBJ_HERO ) {
                const Heroes * hero = world.GetHeroes( { x, y } );
                if ( hero == nullptr ) {
                    continue;
                }
                std::printf( "{\"record\":\"hero\",\"name\":\"%s\",\"color\":%d,\"x\":%d,\"y\":%d,\"attack\":%d,\"defense\":%d,\"army\":[",
                             hero->GetName().c_str(), static_cast<int>( hero->GetColor() ), x, y, hero->GetAttack(), hero->GetDefense() );
                dumpArmy( hero->GetArmy() );
                std::printf( "]}\n" );
            }
            else if ( type == MP2::OBJ_MONSTER ) {
                const Troop troop = Maps::getTroopFromTile( tile );
                std::printf( "{\"record\":\"monster\",\"x\":%d,\"y\":%d,\"monster_id\":%d,\"name\":\"%s\",\"count\":%u}\n", x, y, troop.GetID(),
                             troop.GetName(), troop.GetCount() );
            }
        }

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
            variant.observeObstacles = observePlanes;
            if ( !attackerSpec.empty() && !parseSideSpec( attackerSpec, variant.attacker ) ) {
                std::fprintf( stderr, "cannot parse --attacker %s\n", attackerSpec.c_str() );
                return 2;
            }
            if ( !defenderSpec.empty() && !parseSideSpec( defenderSpec, variant.defender ) ) {
                std::fprintf( stderr, "cannot parse --defender %s\n", defenderSpec.c_str() );
                return 2;
            }
            if ( !attackerHeroSpec.empty() && !parseHeroSpec( attackerHeroSpec, variant.attackerCommander ) ) {
                std::fprintf( stderr, "cannot parse --attacker-hero %s\n", attackerHeroSpec.c_str() );
                return 2;
            }
            if ( !defenderHeroSpec.empty() && !parseHeroSpec( defenderHeroSpec, variant.defenderCommander ) ) {
                std::fprintf( stderr, "cannot parse --defender-hero %s\n", defenderHeroSpec.c_str() );
                return 2;
            }
            variant.allowWideUnits = allowWideUnits;
            const int seedIndex = seedOffset + s;
            if ( seedIndex > 0 ) {
                variant.worldSeed = scenario.worldSeed + static_cast<uint32_t>( seedIndex );
                variant.scenarioId = scenario.scenarioId + "-seed" + std::to_string( seedIndex );
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
            // The controller is constructed after the lambda that reads its probe, so the
            // lambda captures a pointer cell that is filled in right after construction.
            const fheroes2::agent::ExternalDecisionController * probeSource = nullptr;
            auto decide = [&probeSource]( const fheroes2::agent::Observation & observation,
                                          const fheroes2::agent::ActionSet & set ) -> std::optional<uint32_t> {
                std::printf( "{\"record\":\"decision\",\"observation\":%s,\"legal_actions\":[", fheroes2::agent::observationToJson( observation ).c_str() );
                for ( size_t i = 0; i < set.candidates.size(); ++i ) {
                    std::printf( "%s%u", ( i == 0 ) ? "" : ",", set.candidates[i].canonicalIndex );
                }
                std::printf( "]" );
                if ( probeSource != nullptr && probeSource->lastTeacherProbe().has_value() ) {
                    std::printf( ",\"teacher_action\":%u", *probeSource->lastTeacherProbe() );
                }
                std::printf( "}\n" );
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
            if ( observePlanes ) {
                controller.enableObstacleObservation();
            }
            if ( probeTeacher ) {
                controller.enableTeacherProbe();
                probeSource = &controller;
            }
            fheroes2::agent::EpisodeRecording recording;
            recording.auditTeacherCoverage = true;

            std::printf( "{\"record\":\"episode_start\",\"scenario_id\":\"%s\",\"controlled_side\":\"%s\"}\n", scenario.scenarioId.c_str(),
                         controlledSide.c_str() );
            std::fflush( stdout );

            const fheroes2::agent::EpisodeOutcome outcome = fheroes2::agent::runEpisode( scenario, &recording, &controller );

            std::printf( "{\"record\":\"terminal\",\"scenario_id\":\"%s\",\"termination\":\"%s\",\"rounds\":%d"
                         ",\"attacker\":{\"live_stacks\":%u,\"live_creatures\":%u,\"hit_points\":%u,\"strength\":%.3f,\"initial_strength\":%.3f}"
                         ",\"defender\":{\"live_stacks\":%u,\"live_creatures\":%u,\"hit_points\":%u,\"strength\":%.3f,\"initial_strength\":%.3f}"
                         ",\"decisions_seen\":%u,\"decisions_answered\":%u,\"rejected\":%u,\"client_closed\":%s",
                         scenario.scenarioId.c_str(), fheroes2::agent::terminationName( outcome.termination ), outcome.rounds, outcome.attacker.liveStacks,
                         outcome.attacker.liveCreatures, outcome.attacker.hitPoints, outcome.attacker.strength, outcome.attackerInitialStrength,
                         outcome.defender.liveStacks, outcome.defender.liveCreatures, outcome.defender.hitPoints, outcome.defender.strength,
                         outcome.defenderInitialStrength, controller.decisionsSeen(), controller.decisionsAnswered(), controller.rejectedSelections(),
                         controller.isFinished() ? "true" : "false" );
            if ( probeTeacher ) {
                std::printf( ",\"probes_resolved\":%u,\"probes_outside\":%u", controller.probesResolved(), controller.probesOutsideSchema() );
            }
            // The decision-stream digest ("agent_decisions_v0") joins the terminal record so an
            // externally driven episode can be compared with its replay decision by decision,
            // not only at the terminal state; digest inequality names a diverging decision
            // stream even when two battles happen to end alike (#43's investigation tool).
            std::printf( ",\"decision_digest\":\"%s\",\"state_digest\":\"%s\"}\n", recording.decisionDigest.c_str(), outcome.stateDigest.c_str() );
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
