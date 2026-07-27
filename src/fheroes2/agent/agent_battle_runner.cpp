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

#include "agent_battle_runner.h"

#include <cassert>
#include <cstddef>

#include "agent_digest.h"
#include "army.h"
#include "army_troop.h"
#include "battle.h"
#include "battle_arena.h"
#include "battle_army.h"
#include "battle_seed.h"
#include "battle_troop.h"
#include "color.h"
#include "monster.h"
#include "players.h"
#include "race.h"
#include "rand.h"
#include "settings.h"
#include "world.h"

namespace
{
    using fheroes2::agent::EpisodeOutcome;
    using fheroes2::agent::Scenario;
    using fheroes2::agent::SideSummary;
    using fheroes2::agent::StackSpec;
    using fheroes2::agent::Termination;
    using fheroes2::agent::UnitTerminalState;

    void fillArmy( Army & army, const PlayerColor color, const fheroes2::agent::SideSpec & side )
    {
        army.Reset( false );
        army.SetColor( color );

        for ( size_t i = 0; i < side.slots.size(); ++i ) {
            const StackSpec & stack = side.slots[i];
            if ( !stack.isEmpty() ) {
                army.GetTroop( i )->Set( Monster( stack.monsterId ), stack.count );
            }
        }
    }

    void collectForce( const Battle::Force & force, const bool isAttacker, SideSummary & summary, std::vector<UnitTerminalState> & units )
    {
        for ( const Battle::Unit * unit : force ) {
            if ( unit == nullptr ) {
                continue;
            }

            UnitTerminalState state;
            state.uid = unit->GetUID();
            state.monsterId = unit->GetID();
            state.count = unit->GetCount();
            state.hitPoints = unit->GetHitPoints();
            state.headCell = unit->GetHeadIndex();
            state.isAttacker = isAttacker;
            state.isValid = unit->isValid();
            units.push_back( state );

            if ( state.isValid ) {
                ++summary.liveStacks;
                summary.liveCreatures += state.count;
                summary.hitPoints += state.hitPoints;
            }
        }
    }

    std::string computeStateDigest( const EpisodeOutcome & outcome )
    {
        fheroes2::agent::DigestWriter writer;

        writer.appendString( "agent_terminal_v1" );
        writer.appendU32( outcome.effectiveWorldSeed );
        writer.appendU32( outcome.mapSeed );
        writer.appendU32( outcome.combatSeed );
        writer.appendI32( outcome.rounds );
        writer.appendU32( outcome.attackerResult );
        writer.appendU32( outcome.defenderResult );
        writer.appendU8( static_cast<uint8_t>( outcome.termination ) );

        writer.appendU32( static_cast<uint32_t>( outcome.units.size() ) );
        for ( const UnitTerminalState & unit : outcome.units ) {
            writer.appendU32( unit.uid );
            writer.appendI32( unit.monsterId );
            writer.appendU32( unit.count );
            writer.appendU32( unit.hitPoints );
            writer.appendI32( unit.headCell );
            writer.appendU8( unit.isAttacker ? 1 : 0 );
            writer.appendU8( unit.isValid ? 1 : 0 );
        }

        return fheroes2::agent::sha256Hex( writer.bytes() );
    }
}

const char * fheroes2::agent::terminationName( const Termination termination )
{
    switch ( termination ) {
    case Termination::Victory:
        return "victory";
    case Termination::Defeat:
        return "defeat";
    case Termination::EngineDraw:
        return "engine_draw";
    case Termination::RoundLimit:
        return "round_limit";
    default:
        assert( 0 );
        return "unknown";
    }
}

fheroes2::agent::EpisodeOutcome fheroes2::agent::runEpisode( const Scenario & scenario )
{
    assert( validateScenario( scenario ).empty() );

    EpisodeOutcome outcome;
    outcome.effectiveWorldSeed = scenario.worldSeed;

    // Deterministic world generation with zero engine changes: World::Defaults() draws its map
    // seed from the thread-local random device, which is exposed by mutable reference. Keep the
    // reseed immediately before the map generation call -- anything consuming global randomness
    // in between would shift the result (agent spec, section 7.2).
    Rand::CurrentThreadRandomDevice() = Rand::PCG32( scenario.worldSeed );
    world.generateBattleOnlyMap( scenario.groundType );
    outcome.mapSeed = world.GetMapSeed();

    Settings & conf = Settings::Get();
    conf.GetPlayers().Init( static_cast<PlayerColorsSet>( PlayerColor::BLUE ) | static_cast<PlayerColorsSet>( PlayerColor::RED ) );
    world.InitKingdoms();

    Players::SetPlayerRace( PlayerColor::BLUE, Race::KNGT );
    Players::SetPlayerControl( PlayerColor::BLUE, CONTROL_AI );
    Players::SetPlayerRace( PlayerColor::RED, Race::KNGT );
    Players::SetPlayerControl( PlayerColor::RED, CONTROL_AI );

    // Fresh armies per episode; both engine-AI controlled (agent spec, sections 8.2 and 8.3).
    Army attackingArmy;
    fillArmy( attackingArmy, PlayerColor::BLUE, scenario.attacker );
    Army defendingArmy;
    fillArmy( defendingArmy, PlayerColor::RED, scenario.defender );

    outcome.combatSeed = Battle::computeBattleSeed( scenario.tileIndex, outcome.mapSeed, attackingArmy, defendingArmy );

    Rand::PCG32 randomGenerator( outcome.combatSeed );

    {
        // Scoped: the engine allows one arena per process and asserts on a second, so the arena
        // must be destroyed before this function can run again.
        Battle::Arena arena( attackingArmy, defendingArmy, scenario.tileIndex, false, randomGenerator );

        while ( arena.BattleValid() && outcome.rounds < scenario.maxRounds ) {
            arena.Turns();
            ++outcome.rounds;
        }

        const bool truncated = arena.BattleValid();

        const Battle::Result & result = arena.GetResult();
        outcome.attackerResult = result.attacker;
        outcome.defenderResult = result.defender;

        if ( truncated ) {
            outcome.termination = Termination::RoundLimit;
        }
        else if ( ( outcome.attackerResult & Battle::RESULT_WINS ) != 0 ) {
            outcome.termination = Termination::Victory;
        }
        else if ( ( outcome.defenderResult & Battle::RESULT_WINS ) != 0 ) {
            outcome.termination = Termination::Defeat;
        }
        else {
            outcome.termination = Termination::EngineDraw;
        }

        // Terminal state must be read before the arena is destroyed; the input armies are not
        // synchronized after the battle (agent spec, section 8.4).
        collectForce( arena.getAttackingForce(), true, outcome.attacker, outcome.units );
        collectForce( arena.getDefendingForce(), false, outcome.defender, outcome.units );
    }

    outcome.stateDigest = computeStateDigest( outcome );

    return outcome;
}
