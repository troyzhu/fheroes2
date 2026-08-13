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

// Replays a recorded agent episode through the runner, either headless (the default, for
// verifying that the recording reproduces bit-identically) or rendered through the game's own
// battle interface (--render, for watching it). The replay is exact rather than approximate:
// the scenario's world seed regenerates the same map and combat seed, the recorded canonical
// action indices are fed back through the same ExternalDecisionController the worker uses, and
// the terminal state digest printed at the end can be compared across headless and rendered
// runs to prove both played the same battle.
//
// Rendering piggybacks on the game's initializer chain (display, AGG assets, palette) minus
// everything a non-interactive replay does not need: the configuration file is neither read nor
// written, and audio stays uninitialized, which every AudioManager entry point tolerates by
// checking Audio::isValid() first. With --frames-dir, every rendered frame is saved there as a
// numbered BMP plus a manifest line carrying its capture time in milliseconds, through the
// display's generic render observer (the frame-dump logic lives entirely here; the engine only
// offers the null-by-default observer seam). Unlike the play-harness branch's dump, which
// throttles to one overwritten file for a live reader, this keeps every frame, because the
// reader is an encoder running afterwards.
//
// usage: fheroes2_agent_replay --actions FILE [--render] [--frames-dir DIR] [--fixture ID]
//        [--speed 1..10] [--attacker id:count,...] [--defender id:count,...]
//        [--attacker-hero atk:def] [--defender-hero atk:def] [--allow-wide] [--allow-flying]
//        [--side attacker|defender|both]
//
// Play mode (a person battles a checkpoint): --play attacker|defender names the HUMAN side,
// implies --render, and replaces --actions with the worker's line protocol on stdio, so a
// Python wrapper (agent_play/experiments/play_vs.py) answers the other side's decisions from a
// checkpoint while the human plays through the game's own battle interface.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "agent_action_space.h"
#include "agent_battle_runner.h"
#include "agent_external_controller.h"
#include "agent_scenario.h"
#include "agent_trajectory.h"
#include "agg.h"
#include "core.h"
#include "cursor.h"
#include "exception.h"
#include "game.h"
#include "game_assets.h"
#include "h2d.h"
#include "icn.h"
#include "image_palette.h"
#include "image_tool.h"
#include "localevent.h"
#include "render_processor.h"
#include "screen.h"
#include "settings.h"
#include "ui_tool.h"

namespace
{
    // Same dialect as src/agent_worker/main.cpp ("1:50,2:10" -> Peasant x50, Archer x10), kept
    // in lockstep so a recording captured against the worker replays under identical parsing.
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

    // Saves every rendered frame as a numbered BMP and appends its capture time to a manifest,
    // so the video encoder can reproduce the engine's real animation cadence afterwards.
    fheroes2::Display::RenderObserver makeFrameDumper( const std::string & framesDir )
    {
        return [framesDir, frameCounter = uint32_t{ 0 },
                firstFrameTime = std::chrono::steady_clock::now()]( const fheroes2::Display & display ) mutable {
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - firstFrameTime ).count();

            char frameName[32];
            std::snprintf( frameName, sizeof( frameName ), "frame_%06u.bmp", frameCounter );

            if ( !fheroes2::Save( display, framesDir + "/" + frameName ) ) {
                return;
            }

            std::FILE * manifest = std::fopen( ( framesDir + "/manifest.tsv" ).c_str(), "a" );
            if ( manifest != nullptr ) {
                std::fprintf( manifest, "%s\t%lld\n", frameName, static_cast<long long>( elapsedMs ) );
                std::fclose( manifest );
            }

            ++frameCounter;
        };
    }

    // One canonical action index per line, in decision order, as capture_replay.py records them.
    std::optional<std::vector<uint32_t>> readActions( const std::string & path )
    {
        std::ifstream input( path );
        if ( !input ) {
            return std::nullopt;
        }
        std::vector<uint32_t> actions;
        std::string line;
        while ( std::getline( input, line ) ) {
            if ( line.empty() ) {
                continue;
            }
            try {
                actions.push_back( static_cast<uint32_t>( std::stoul( line ) ) );
            }
            catch ( ... ) {
                return std::nullopt;
            }
        }
        return actions;
    }
}

namespace
{
    // The whole tool, so main can guarantee no exception ever escapes: an uncaught one aborts
    // the process, which macOS reports to the user as a crash even when the battle ended
    // normally. Interactive play reaches several engine paths that signal by exception.
    int runTool( int argc, char ** argv );
}

int main( int argc, char ** argv )
{
    try {
        return runTool( argc, argv );
    }
    catch ( const fheroes2::UserRequestedApplicationClosure & ) {
        std::fprintf( stderr, "closed by the user\n" );
        return 4;
    }
    catch ( const std::exception & error ) {
        std::fprintf( stderr, "error: %s\n", error.what() );
        return 5;
    }
    catch ( ... ) {
        std::fprintf( stderr, "unknown error\n" );
        return 5;
    }
}

namespace
{
int runTool( int argc, char ** argv )
{
    std::string actionsPath;
    std::string fixtureId = "m1_tiny_melee";
    // Which world seed the recording was made under. The worker derives the obstacle layout from
    // it, so a replay of a battle fought on another variant reaches a different board and the
    // recorded action stream stops matching part-way through.
    int seedOffset = 0;
    std::string attackerSpec;
    std::string defenderSpec;
    std::string attackerHeroSpec;
    std::string defenderHeroSpec;
    std::string controlledSide = "attacker";
    std::string framesDir;
    std::string playSide;
    bool allowWideUnits = false;
    // flying_v1, off by default so existing scenarios and golden digests are untouched.
    bool allowFlyingUnits = false;
    bool render = false;
    int battleSpeed = 10;

    for ( int i = 1; i < argc; ++i ) {
        const auto value = [&]( const char * flag ) -> const char * {
            if ( i + 1 >= argc ) {
                std::fprintf( stderr, "%s needs a value\n", flag );
                std::exit( 2 );
            }
            return argv[++i];
        };

        if ( std::strcmp( argv[i], "--actions" ) == 0 ) {
            actionsPath = value( "--actions" );
        }
        else if ( std::strcmp( argv[i], "--fixture" ) == 0 ) {
            fixtureId = value( "--fixture" );
        }
        else if ( std::strcmp( argv[i], "--attacker" ) == 0 ) {
            attackerSpec = value( "--attacker" );
        }
        else if ( std::strcmp( argv[i], "--defender" ) == 0 ) {
            defenderSpec = value( "--defender" );
        }
        else if ( std::strcmp( argv[i], "--attacker-hero" ) == 0 ) {
            attackerHeroSpec = value( "--attacker-hero" );
        }
        else if ( std::strcmp( argv[i], "--defender-hero" ) == 0 ) {
            defenderHeroSpec = value( "--defender-hero" );
        }
        else if ( std::strcmp( argv[i], "--side" ) == 0 ) {
            controlledSide = value( "--side" );
        }
        else if ( std::strcmp( argv[i], "--speed" ) == 0 ) {
            battleSpeed = std::atoi( value( "--speed" ) );
        }
        else if ( std::strcmp( argv[i], "--frames-dir" ) == 0 ) {
            framesDir = value( "--frames-dir" );
        }
        else if ( std::strcmp( argv[i], "--seed-offset" ) == 0 ) {
            seedOffset = std::atoi( value( "--seed-offset" ) );
        }
        else if ( std::strcmp( argv[i], "--allow-flying" ) == 0 ) {
            allowFlyingUnits = true;
        }
        else if ( std::strcmp( argv[i], "--allow-wide" ) == 0 ) {
            allowWideUnits = true;
        }
        else if ( std::strcmp( argv[i], "--render" ) == 0 ) {
            render = true;
        }
        else if ( std::strcmp( argv[i], "--play" ) == 0 ) {
            playSide = value( "--play" );
            render = true;
        }
        else {
            std::fprintf( stderr,
                          "usage: fheroes2_agent_replay --actions FILE [--render] [--frames-dir DIR] [--fixture ID]\n"
                          "       [--speed 1..10] [--attacker id:count,...] [--defender id:count,...]\n"
                          "       [--attacker-hero atk:def] [--defender-hero atk:def] [--allow-wide] [--seed-offset N]\n"
                          "       [--side attacker|defender|both] [--allow-flying] [--play attacker|defender]\n" );
            return 2;
        }
    }

    if ( !framesDir.empty() && !render ) {
        std::fprintf( stderr, "--frames-dir needs --render\n" );
        return 2;
    }

    if ( !playSide.empty() && playSide != "attacker" && playSide != "defender" ) {
        std::fprintf( stderr, "--play must be attacker or defender\n" );
        return 2;
    }
    if ( actionsPath.empty() && playSide.empty() ) {
        std::fprintf( stderr, "--actions is required (or --play for a live game)\n" );
        return 2;
    }
    std::optional<std::vector<uint32_t>> actions;
    if ( !actionsPath.empty() ) {
        actions = readActions( actionsPath );
        if ( !actions.has_value() || actions->empty() ) {
            std::fprintf( stderr, "cannot read actions from %s\n", actionsPath.c_str() );
            return 2;
        }
    }

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

    fheroes2::agent::Scenario scenario;
    bool fixtureFound = false;
    for ( const auto & fixture : fheroes2::agent::milestone1Fixtures() ) {
        if ( fixture.scenarioId == fixtureId ) {
            scenario = fixture;
            fixtureFound = true;
            break;
        }
    }
    if ( !fixtureFound ) {
        std::fprintf( stderr, "unknown fixture %s\n", fixtureId.c_str() );
        return 2;
    }

    if ( !attackerSpec.empty() && !parseSideSpec( attackerSpec, scenario.attacker ) ) {
        std::fprintf( stderr, "cannot parse --attacker %s\n", attackerSpec.c_str() );
        return 2;
    }
    if ( !defenderSpec.empty() && !parseSideSpec( defenderSpec, scenario.defender ) ) {
        std::fprintf( stderr, "cannot parse --defender %s\n", defenderSpec.c_str() );
        return 2;
    }
    if ( !attackerHeroSpec.empty() && !parseHeroSpec( attackerHeroSpec, scenario.attackerCommander ) ) {
        std::fprintf( stderr, "cannot parse --attacker-hero %s\n", attackerHeroSpec.c_str() );
        return 2;
    }
    if ( !defenderHeroSpec.empty() && !parseHeroSpec( defenderHeroSpec, scenario.defenderCommander ) ) {
        std::fprintf( stderr, "cannot parse --defender-hero %s\n", defenderHeroSpec.c_str() );
        return 2;
    }
    scenario.allowWideUnits = allowWideUnits;
    scenario.allowFlyingUnits = allowFlyingUnits;
    // Matches the worker's own rule in src/agent_worker/main.cpp: seed index zero keeps the
    // fixture's seed untouched, so recordings made before --seed-offset existed replay unchanged.
    if ( seedOffset > 0 ) {
        scenario.worldSeed = scenario.worldSeed + static_cast<uint32_t>( seedOffset );
        scenario.scenarioId = scenario.scenarioId + "-seed" + std::to_string( seedOffset );
    }

    const std::string problem = fheroes2::agent::validateScenario( scenario );
    if ( !problem.empty() ) {
        std::fprintf( stderr, "invalid scenario: %s\n", problem.c_str() );
        return 2;
    }

    Settings & conf = Settings::Get();
    conf.SetProgramPath( argv[0] );

    // Rendering prerequisites, alive until the episode finishes. The order mirrors the game's
    // own main(): hardware and SDL core, then the display, then the assets under a palette
    // restorer, then the game palette and the animation delay tables via Game::Init().
    std::optional<System::HardwareInitializer> hardwareInitializer;
    std::optional<System::CoreInitializer> coreInitializer;
    std::optional<AGG::AGGInitializer> aggInitializer;
    std::optional<fheroes2::h2d::H2DInitializer> h2dInitializer;

    // Teardown in the game's own order, which is load-bearing: the asset initializers go first
    // and the display is released only afterwards, because releasing it out from under them
    // aborts during their destruction, and a throwing destructor cannot be caught anywhere.
    const auto shutdownRender = [&aggInitializer, &h2dInitializer]() {
        h2dInitializer.reset();
        aggInitializer.reset();
        fheroes2::Display::instance().setRenderObserver( {} );
        fheroes2::RenderProcessor::instance().unregisterRenderers();
        fheroes2::Display::instance().release();
    };

    if ( render ) {
        hardwareInitializer.emplace();
        coreInitializer.emplace();

        fheroes2::Display & display = fheroes2::Display::instance();
        display.setResolution( fheroes2::ResolutionInfo( fheroes2::Display::DEFAULT_WIDTH, fheroes2::Display::DEFAULT_HEIGHT ) );
        fheroes2::engine().setTitle( playSide.empty() ? "fheroes2 agent replay" : "fheroes2: you versus the agent" );

        auto & cursor = fheroes2::cursor();
        if ( playSide.empty() ) {
            // A recorded replay has no one to click, and the game's own initializer hides the
            // cursor at startup too.
            cursor.show( false );
        }
        else {
            // Interactive play needs what the game's DisplayInitializer sets up for it: the
            // configured emulation mode, the updater that redraws the cursor when the theme
            // changes, and a visible cursor. Without these the battle renders but no pointer
            // ever appears, so nothing can be clicked.
            cursor.enableSoftwareEmulation( conf.isSoftwareEmulationEnabled() );
            cursor.registerUpdater( Cursor::Refresh );
            cursor.show( true );
        }

        fheroes2::RenderProcessor & renderProcessor = fheroes2::RenderProcessor::instance();
        display.subscribe( [&renderProcessor]( std::vector<uint8_t> & palette ) { return renderProcessor.preRenderAction( palette ); },
                           [&renderProcessor]() { renderProcessor.postRenderAction(); } );
        renderProcessor.startColorCycling();

        {
            const fheroes2::ScreenPaletteRestorer screenRestorer;
            aggInitializer.emplace();
            h2dInitializer.emplace();
            Assets::getImage( ICN::FONT, 0 );
        }

        fheroes2::setGamePalette( AGG::getDataFromAggFile( "KB.PAL", false ) );
        display.changePalette( nullptr, true );

        // Audio stays uninitialized, so engine sound calls no-op; the volumes are zeroed as well
        // so no future code path can make a rendered replay audible. A render window on someone's
        // desk must be silent.
        conf.SetSoundVolume( 0 );
        conf.SetMusicVolume( 0 );

        conf.setGameLanguage( conf.getGameLanguage() );
        Game::Init();

        // Game::Init installs a quit hook that asks the engine to throw on window close, and the
        // engine processes the pending quit again while the battle interface tears down, so the
        // second throw lands during unwinding where no catch can exist and the process aborts,
        // which macOS reports as a crash. This hook throws exactly once: the first close ends the
        // battle through the catch below, and every later quit event is ignored.
        LocalEvent::Get().setQuitEventProcessingHook( []() {
            static bool alreadyRequested = false;
            if ( alreadyRequested ) {
                return false;
            }
            alreadyRequested = true;
            return true;
        } );

        if ( !framesDir.empty() ) {
            display.setRenderObserver( makeFrameDumper( framesDir ) );
        }
    }

    conf.SetBattleSpeed( battleSpeed );

    size_t nextAction = 0;
    fheroes2::agent::ExternalDecisionController::DecideFn decide;
    fheroes2::agent::HumanSide humanSide = fheroes2::agent::HumanSide::None;
    if ( playSide.empty() ) {
        decide = [&actions, &nextAction]( const fheroes2::agent::Observation & /* observation */,
                                          const fheroes2::agent::ActionSet & /* set */ ) -> std::optional<uint32_t> {
            if ( nextAction >= actions->size() ) {
                // Past the end of the recording: the live battle asked for more decisions than
                // the recorded one had, which is divergence. Decline, so the controller unwinds.
                return std::nullopt;
            }
            return ( *actions )[nextAction++];
        };
    }
    else {
        // Play mode: the human takes playSide through the interface, the controller takes the
        // other side, and its decisions travel the worker's line protocol on stdio so a Python
        // wrapper can answer them from a checkpoint.
        humanSide = ( playSide == "attacker" ) ? fheroes2::agent::HumanSide::Attacker : fheroes2::agent::HumanSide::Defender;
        side = ( playSide == "attacker" ) ? fheroes2::agent::ControlledSide::Defender : fheroes2::agent::ControlledSide::Attacker;
        decide = []( const fheroes2::agent::Observation & observation,
                     const fheroes2::agent::ActionSet & set ) -> std::optional<uint32_t> {
            std::printf( "{\"record\":\"decision\",\"observation\":%s,\"legal_actions\":[", fheroes2::agent::observationToJson( observation ).c_str() );
            for ( size_t i = 0; i < set.candidates.size(); ++i ) {
                std::printf( "%s%u", ( i == 0 ) ? "" : ",", set.candidates[i].canonicalIndex );
            }
            std::printf( "]}\n" );
            std::fflush( stdout );
            std::string line;
            if ( !std::getline( std::cin, line ) ) {
                return std::nullopt;
            }
            try {
                return static_cast<uint32_t>( std::stoul( line ) );
            }
            catch ( ... ) {
                return std::nullopt;
            }
        };
    }

    fheroes2::agent::ExternalDecisionController controller( side, decide );

    // Wired exactly as the worker's protocol mode wires its runEpisode call (a passive
    // recording with the coverage audit on), so a replayed episode and a protocol-driven one
    // run byte-identical hook plumbing, and the decision-stream digest printed below compares
    // the two paths decision by decision rather than only at the terminal state. That
    // per-decision comparison is the #43 investigation tool. Interactive play keeps the
    // historical no-recording call: a human's decisions need no replay digest.
    fheroes2::agent::EpisodeRecording recording;
    recording.auditTeacherCoverage = true;

    fheroes2::agent::EpisodeOutcome outcome;
    try {
        outcome = fheroes2::agent::runEpisode( scenario, playSide.empty() ? &recording : nullptr, &controller, render, humanSide );
    }
    catch ( const fheroes2::UserRequestedApplicationClosure & ) {
        // Closing the battle window is a normal way to end a live game, and the engine reports it
        // by exception. Returning normally from here still aborted: something in the render
        // teardown or a static destructor throws afterwards, and a throwing destructor is
        // uncatchable by construction, so macOS kept reporting a crash on a clean quit. This tool
        // owns no state worth persisting past this point, so it flushes and leaves immediately
        // instead of unwinding. The operating system reclaims the window, the audio device and
        // the process memory.
        std::fprintf( stderr, "battle window closed by the user\n" );
        std::fflush( nullptr );
        std::_Exit( 4 );
    }

    if ( render ) {
        shutdownRender();
    }

    const bool exact = playSide.empty() && ( controller.rejectedSelections() == 0 ) && !controller.isFinished()
                       && ( nextAction == actions->size() );
    std::printf( "{\"record\":\"replay_terminal\",\"scenario_id\":\"%s\",\"termination\":\"%s\",\"rounds\":%d"
                 ",\"actions_recorded\":%zu,\"actions_used\":%zu,\"decisions_seen\":%u,\"rejected\":%u,\"exact\":%s"
                 ",\"attacker\":{\"live_stacks\":%u,\"live_creatures\":%u,\"hit_points\":%u}"
                 ",\"defender\":{\"live_stacks\":%u,\"live_creatures\":%u,\"hit_points\":%u}"
                 ",\"decision_digest\":\"%s\",\"state_digest\":\"%s\"}\n",
                 scenario.scenarioId.c_str(), fheroes2::agent::terminationName( outcome.termination ), outcome.rounds,
                 actions.has_value() ? actions->size() : static_cast<size_t>( 0 ), nextAction,
                 controller.decisionsSeen(), controller.rejectedSelections(), exact ? "true" : "false", outcome.attacker.liveStacks, outcome.attacker.liveCreatures,
                 outcome.attacker.hitPoints, outcome.defender.liveStacks, outcome.defender.liveCreatures, outcome.defender.hitPoints, recording.decisionDigest.c_str(),
                 outcome.stateDigest.c_str() );
    std::fflush( stdout );

    if ( !playSide.empty() ) {
        // Same reasoning as the closure path above: a finished live battle has printed its
        // result, and unwinding the interactive render stack is what produced the crash report.
        std::fflush( nullptr );
        std::_Exit( exact ? 0 : 3 );
    }

    return exact ? 0 : 3;
}
}
