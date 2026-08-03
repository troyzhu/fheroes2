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

#include "agent_observation.h"

#include <algorithm>
#include <string>

#include "battle_arena.h"
#include "battle_army.h"
#include "battle_troop.h"

namespace
{
    void collectSide( const Battle::Force & force, const bool isAttacker, const uint32_t activeUid, std::vector<fheroes2::agent::ObservedUnit> & out )
    {
        for ( const Battle::Unit * unit : force ) {
            // A dead stack is off the board, so it is not part of what a player sees
            // (agent spec section 12.3, "serialize living battle units only").
            if ( unit == nullptr || !unit->isValid() ) {
                continue;
            }

            fheroes2::agent::ObservedUnit observed;
            observed.uid = unit->GetUID();
            observed.monsterId = unit->GetID();
            observed.isAttacker = isAttacker;
            observed.isActive = ( unit->GetUID() == activeUid );
            observed.count = unit->GetCount();
            observed.initialCount = unit->GetInitialCount();
            observed.hitPoints = unit->GetHitPoints();
            observed.topHitPoints = unit->GetHitPointsLeft();
            observed.attack = unit->GetAttack();
            observed.defense = unit->GetDefense();
            observed.speed = unit->GetSpeed();
            observed.shots = unit->GetShots();
            observed.morale = unit->GetMorale();
            observed.luck = unit->GetLuck();
            observed.headCell = unit->GetHeadIndex();
            observed.tailCell = unit->isWide() ? unit->GetTailIndex() : -1;
            observed.isWide = unit->isWide();
            observed.isFlying = unit->isFlying();
            observed.isArcher = unit->isArchers();
            observed.isHandFighting = unit->isHandFighting();
            out.push_back( observed );
        }
    }

    void appendBool( std::string & json, const char * name, const bool value )
    {
        json += ",\"";
        json += name;
        json += "\":";
        json += value ? "true" : "false";
    }

    void appendInt( std::string & json, const char * name, const long long value )
    {
        json += ",\"";
        json += name;
        json += "\":";
        json += std::to_string( value );
    }
}

fheroes2::agent::Observation fheroes2::agent::captureObservation( const Battle::Arena & arena, const Battle::Unit & currentUnit )
{
    Observation observation;
    observation.engineDecisionIndex = arena.GetEngineDecisionIndex();
    observation.round = arena.GetTurnNumber();
    observation.activeUid = currentUnit.GetUID();

    collectSide( arena.getAttackingForce(), true, observation.activeUid, observation.units );
    collectSide( arena.getDefendingForce(), false, observation.activeUid, observation.units );

    // Engine container order is an implementation detail. Sorting by uid makes the observation
    // a function of state alone, which is what lets two runs be compared byte for byte.
    std::sort( observation.units.begin(), observation.units.end(), []( const ObservedUnit & a, const ObservedUnit & b ) { return a.uid < b.uid; } );

    for ( const ObservedUnit & unit : observation.units ) {
        if ( unit.isActive ) {
            observation.activeIsAttacker = unit.isAttacker;
            break;
        }
    }

    return observation;
}

std::string fheroes2::agent::observationToJson( const Observation & observation )
{
    std::string json = "{\"schema\":\"observation_full_v1\"";
    appendInt( json, "engine_decision_index", observation.engineDecisionIndex );
    appendInt( json, "round", observation.round );
    appendInt( json, "active_uid", observation.activeUid );
    appendBool( json, "active_is_attacker", observation.activeIsAttacker );

    json += ",\"units\":[";
    for ( size_t i = 0; i < observation.units.size(); ++i ) {
        const ObservedUnit & unit = observation.units[i];
        if ( i != 0 ) {
            json += ',';
        }
        json += "{\"uid\":";
        json += std::to_string( unit.uid );
        appendInt( json, "monster_id", unit.monsterId );
        json += ",\"side\":\"";
        json += unit.isAttacker ? "attacker" : "defender";
        json += '"';
        appendBool( json, "active", unit.isActive );
        appendInt( json, "count", unit.count );
        appendInt( json, "initial_count", unit.initialCount );
        appendInt( json, "hit_points", unit.hitPoints );
        appendInt( json, "top_hit_points", unit.topHitPoints );
        appendInt( json, "attack", unit.attack );
        appendInt( json, "defense", unit.defense );
        appendInt( json, "speed", unit.speed );
        appendInt( json, "shots", unit.shots );
        appendInt( json, "morale", unit.morale );
        appendInt( json, "luck", unit.luck );
        appendInt( json, "head_cell", unit.headCell );
        appendInt( json, "tail_cell", unit.tailCell );
        appendBool( json, "wide", unit.isWide );
        appendBool( json, "flying", unit.isFlying );
        appendBool( json, "archer", unit.isArcher );
        appendBool( json, "hand_fighting", unit.isHandFighting );
        json += '}';
    }
    json += "]}";
    return json;
}
