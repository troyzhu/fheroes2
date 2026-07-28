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

#include "agent_command_snapshot.h"

#include <cassert>
#include <cstddef>

fheroes2::agent::CommandSnapshot fheroes2::agent::snapshotCommand( const Battle::Command & command )
{
    CommandSnapshot snapshot;
    snapshot.type = command.GetType();

    // Command derives from std::vector<int>, so this copy is exact; GetNextValue() mutates the
    // copy only, popping parameters from the back — i.e. yielding them in semantic order.
    Battle::Command scratch = command;

    const size_t parameterCount = scratch.size();
    snapshot.params.reserve( parameterCount );
    for ( size_t i = 0; i < parameterCount; ++i ) {
        snapshot.params.push_back( scratch.GetNextValue() );
    }

    switch ( snapshot.type ) {
    case Battle::CommandType::MOVE:
        assert( snapshot.params.size() == 2 );
        if ( snapshot.params.size() == 2 ) {
            snapshot.unitUid = static_cast<uint32_t>( snapshot.params[0] );
            snapshot.moveCell = snapshot.params[1];
        }
        break;
    case Battle::CommandType::ATTACK:
        assert( snapshot.params.size() == 5 );
        if ( snapshot.params.size() == 5 ) {
            snapshot.unitUid = static_cast<uint32_t>( snapshot.params[0] );
            snapshot.defenderUid = static_cast<uint32_t>( snapshot.params[1] );
            snapshot.moveCell = snapshot.params[2];
            snapshot.targetCell = snapshot.params[3];
            snapshot.direction = snapshot.params[4];
        }
        break;
    case Battle::CommandType::SKIP:
        assert( snapshot.params.size() == 1 );
        if ( snapshot.params.size() == 1 ) {
            snapshot.unitUid = static_cast<uint32_t>( snapshot.params[0] );
        }
        break;
    case Battle::CommandType::MORALE:
        assert( snapshot.params.size() == 2 );
        if ( snapshot.params.size() == 2 ) {
            snapshot.unitUid = static_cast<uint32_t>( snapshot.params[0] );
            snapshot.moraleIsGood = ( snapshot.params[1] != 0 );
        }
        break;
    default:
        // Other command types (spells, siege machinery, flow control) cannot occur in
        // creature-only field battles; `params` still captures them faithfully.
        break;
    }

    return snapshot;
}

const char * fheroes2::agent::commandTypeName( const Battle::CommandType type )
{
    switch ( type ) {
    case Battle::CommandType::MOVE:
        return "move";
    case Battle::CommandType::ATTACK:
        return "attack";
    case Battle::CommandType::SPELLCAST:
        return "spellcast";
    case Battle::CommandType::MORALE:
        return "morale";
    case Battle::CommandType::CATAPULT:
        return "catapult";
    case Battle::CommandType::TOWER:
        return "tower";
    case Battle::CommandType::RETREAT:
        return "retreat";
    case Battle::CommandType::SURRENDER:
        return "surrender";
    case Battle::CommandType::SKIP:
        return "skip";
    case Battle::CommandType::TOGGLE_AUTO_COMBAT:
        return "toggle_auto_combat";
    case Battle::CommandType::QUICK_COMBAT:
        return "quick_combat";
    default:
        assert( 0 );
        return "unknown";
    }
}

std::string fheroes2::agent::canonicalCommandKey( const CommandSnapshot & snapshot )
{
    std::string key = commandTypeName( snapshot.type );
    for ( const int param : snapshot.params ) {
        key += ':';
        key += std::to_string( param );
    }
    return key;
}
