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

#include "agent_scenario.h"

#include "agent_capabilities.h"
#include "army.h"
#include "ground.h"
#include "monster.h"

namespace
{
    using fheroes2::agent::Scenario;
    using fheroes2::agent::scenarioMaxStackCount;
    using fheroes2::agent::SideSpec;
    using fheroes2::agent::StackSpec;

    static_assert( fheroes2::agent::scenarioSlotCount == Army::maximumTroopCount, "Scenario slot count must match the engine army slot count" );

    std::string validateSide( const std::string & scenarioId, const char * sideName, const SideSpec & side )
    {
        size_t liveSlots = 0;

        for ( size_t i = 0; i < side.slots.size(); ++i ) {
            const StackSpec & stack = side.slots[i];
            if ( stack.isEmpty() ) {
                continue;
            }

            ++liveSlots;

            const fheroes2::agent::MonsterCapability capability = fheroes2::agent::auditMonster( stack.monsterId );
            if ( !capability.isValid ) {
                return "scenario '" + scenarioId + "': " + sideName + " slot " + std::to_string( i ) + " has invalid monster id "
                       + std::to_string( stack.monsterId );
            }
            // Profile gate (spec 11.1): only creatures whose action space simple_v1 fully
            // covers may appear on either side.
            if ( !capability.simpleV1Supported ) {
                return "scenario '" + scenarioId + "': " + sideName + " slot " + std::to_string( i ) + " monster " + capability.name
                       + " is not supported by simple_v1 (" + capability.reason + ")";
            }
            if ( stack.count > scenarioMaxStackCount ) {
                return "scenario '" + scenarioId + "': " + sideName + " slot " + std::to_string( i ) + " count " + std::to_string( stack.count )
                       + " exceeds the safety maximum " + std::to_string( scenarioMaxStackCount );
            }
        }

        if ( liveSlots == 0 ) {
            return "scenario '" + scenarioId + "': " + sideName + " has no valid stacks";
        }

        return {};
    }
}

std::string fheroes2::agent::validateScenario( const Scenario & scenario )
{
    if ( scenario.scenarioId.empty() ) {
        return "scenario id must not be empty";
    }

    if ( scenario.tileIndex != 1 ) {
        return "scenario '" + scenario.scenarioId + "': tile index must be 1 (got " + std::to_string( scenario.tileIndex ) + ")";
    }

    const int ground = scenario.groundType;
    const bool isSingleGroundBit = ( ground != 0 ) && ( ( ground & ( ground - 1 ) ) == 0 );
    if ( ground == Maps::Ground::UNKNOWN || !isSingleGroundBit || ( ground & Maps::Ground::ALL ) != ground ) {
        return "scenario '" + scenario.scenarioId + "': ground type " + std::to_string( ground ) + " is not a single supported Maps::Ground value";
    }

    if ( scenario.maxRounds < 1 || scenario.maxRounds > 10000 ) {
        return "scenario '" + scenario.scenarioId + "': max rounds " + std::to_string( scenario.maxRounds ) + " is outside [1, 10000]";
    }

    std::string sideError = validateSide( scenario.scenarioId, "attacker", scenario.attacker );
    if ( !sideError.empty() ) {
        return sideError;
    }
    sideError = validateSide( scenario.scenarioId, "defender", scenario.defender );
    if ( !sideError.empty() ) {
        return sideError;
    }

    const auto validateCommander = [&scenario]( const char * side, const CommanderSpec & commander ) -> std::string {
        if ( !commander.present ) {
            // An absent commander must carry no stats, or a caller has set stats and forgotten
            // the flag, which would silently run the battle without them.
            if ( commander.attack != 0 || commander.defense != 0 ) {
                return "scenario '" + scenario.scenarioId + "': " + side + " commander stats set but commander not marked present";
            }
            return {};
        }
        if ( commander.attack < 0 || commander.attack > scenarioMaxCommanderStat || commander.defense < 0 || commander.defense > scenarioMaxCommanderStat ) {
            return "scenario '" + scenario.scenarioId + "': " + side + " commander stats (" + std::to_string( commander.attack ) + ", "
                   + std::to_string( commander.defense ) + ") are outside [0, " + std::to_string( scenarioMaxCommanderStat ) + "]";
        }
        return {};
    };

    std::string commanderError = validateCommander( "attacker", scenario.attackerCommander );
    if ( !commanderError.empty() ) {
        return commanderError;
    }
    commanderError = validateCommander( "defender", scenario.defenderCommander );
    if ( !commanderError.empty() ) {
        return commanderError;
    }

    return {};
}

const std::vector<fheroes2::agent::Scenario> & fheroes2::agent::milestone1Fixtures()
{
    static const std::vector<Scenario> fixtures = []() {
        std::vector<Scenario> result;

        {
            // Same composition, terrain and world seed as the Phase 0 spike default, so the
            // fixture's map and combat seeds cross-check the historical baseline.
            Scenario s;
            s.scenarioId = "m1_tiny_melee";
            s.groundType = Maps::Ground::GRASS;
            s.worldSeed = 20260726;
            s.attacker.slots[0] = { Monster::PEASANT, 50 };
            s.defender.slots[0] = { Monster::PEASANT, 50 };
            result.push_back( s );
        }

        {
            Scenario s;
            s.scenarioId = "m1_three_stack";
            s.groundType = Maps::Ground::DIRT;
            s.worldSeed = 20260727;
            s.attacker.slots[0] = { Monster::PEASANT, 30 };
            s.attacker.slots[1] = { Monster::ARCHER, 15 };
            s.attacker.slots[2] = { Monster::PEASANT, 30 };
            s.defender.slots[0] = { Monster::PEASANT, 40 };
            s.defender.slots[1] = { Monster::RANGER, 10 };
            s.defender.slots[2] = { Monster::PEASANT, 20 };
            result.push_back( s );
        }

        {
            // Interior empty slots on purpose: slot position is part of the combat seed.
            Scenario s;
            s.scenarioId = "m1_five_stack";
            s.groundType = Maps::Ground::GRASS;
            s.worldSeed = 20260728;
            s.attacker.slots[0] = { Monster::PEASANT, 25 };
            s.attacker.slots[1] = { Monster::ARCHER, 12 };
            s.attacker.slots[2] = { Monster::PEASANT, 25 };
            s.attacker.slots[3] = { Monster::RANGER, 8 };
            s.attacker.slots[4] = { Monster::PEASANT, 25 };
            s.defender.slots[0] = { Monster::PEASANT, 35 };
            s.defender.slots[1] = { Monster::PEASANT, 35 };
            s.defender.slots[2] = { Monster::ARCHER, 10 };
            s.defender.slots[3] = { Monster::PEASANT, 35 };
            s.defender.slots[4] = { Monster::RANGER, 6 };
            result.push_back( s );
        }

        {
            Scenario s;
            s.scenarioId = "m1_ranged_heavy";
            s.groundType = Maps::Ground::WASTELAND;
            s.worldSeed = 20260729;
            s.attacker.slots[0] = { Monster::ARCHER, 25 };
            s.attacker.slots[1] = { Monster::RANGER, 15 };
            s.attacker.slots[2] = { Monster::ARCHER, 25 };
            s.defender.slots[0] = { Monster::PEASANT, 120 };
            s.defender.slots[1] = { Monster::PEASANT, 80 };
            result.push_back( s );
        }

        {
            // Larger mixed armies on both sides to keep the battle going for multiple rounds --
            // the closest Milestone 1 gets to the "longer balanced battle" workload shape.
            Scenario s;
            s.scenarioId = "m1_longer_balanced";
            s.groundType = Maps::Ground::SWAMP;
            s.worldSeed = 20260730;
            s.attacker.slots[0] = { Monster::PEASANT, 80 };
            s.attacker.slots[1] = { Monster::RANGER, 20 };
            s.attacker.slots[2] = { Monster::PEASANT, 80 };
            s.attacker.slots[3] = { Monster::ARCHER, 20 };
            s.attacker.slots[4] = { Monster::PEASANT, 80 };
            s.defender.slots[0] = { Monster::PEASANT, 90 };
            s.defender.slots[1] = { Monster::ARCHER, 18 };
            s.defender.slots[2] = { Monster::PEASANT, 70 };
            s.defender.slots[3] = { Monster::RANGER, 16 };
            s.defender.slots[4] = { Monster::PEASANT, 90 };
            result.push_back( s );
        }

        return result;
    }();

    return fixtures;
}
