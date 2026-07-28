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

#include <cstdint>
#include <string>
#include <vector>

#include "battle_command.h"

namespace fheroes2::agent
{
    // Typed, decoded view of one Battle::Command (agent spec, section 10.5).
    //
    // Battle::Command stores its integer parameters in reverse vector order and consumes them
    // with GetNextValue(), which pops from the back — so a snapshot must never be built by
    // iterating the raw vector. snapshotCommand() decodes a copy; the original command is left
    // untouched and can still be applied by the engine.
    struct CommandSnapshot
    {
        Battle::CommandType type{ Battle::CommandType::SKIP };

        // Every integer parameter in semantic order (the order documented in the Command
        // constructor), regardless of the command type.
        std::vector<int> params;

        // Decoded fields for the command types that occur in creature-only battles.
        // Untouched defaults for other types; `params` is always authoritative.
        uint32_t unitUid{ 0 };      // MOVE / SKIP / MORALE unit, ATTACK attacker
        uint32_t defenderUid{ 0 };  // ATTACK
        int32_t moveCell{ -1 };     // MOVE destination, ATTACK movement cell (-1: attack in place)
        int32_t targetCell{ -1 };   // ATTACK
        int32_t direction{ 0 };     // ATTACK
        bool moraleIsGood{ false }; // MORALE
    };

    // Decodes a copy of the command with GetNextValue(); the original is not consumed.
    CommandSnapshot snapshotCommand( const Battle::Command & command );

    // Stable lowercase name for a command type ("move", "attack", ...).
    const char * commandTypeName( const Battle::CommandType type );

    // Canonical semantic key, e.g. "move:7:34", "attack:1:6:34:45:3", "skip:9", "morale:5:1".
    // Unhandled types render as "<name>:p0:p1:...". Part of the trajectory data contract.
    std::string canonicalCommandKey( const CommandSnapshot & snapshot );
}
