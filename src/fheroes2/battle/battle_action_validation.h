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
#include <optional>

#include "battle_cell.h"

namespace Battle
{
    class Unit;

    // The exact legality rules used by Arena::ApplyAction{Move,Attack,Skip}, extracted so that
    // command execution and external candidate enumeration share one implementation instead of
    // re-deriving battle rules (the tactical AI and the human interface each carry their own
    // near-duplicates of this logic; do not add another).
    //
    // All functions here are silent predicates/resolvers: they never mutate game state, never
    // consume combat randomness, and never log. They may warm the battle pathfinder's cache
    // through Position::GetReachable, which is deterministic. They implicitly evaluate against
    // the process-global arena, like the Board/Position primitives they are built on.

    // Computes which cell of the defender gets hit when attacking from the given position
    // (head cell first, then tail); falls back to the defender's head cell, which callers
    // interpret as "most likely a shot".
    int32_t calculateAttackTarget( const Unit & attackingUnit, const Position & attackPosition, const Unit & defendingUnit );

    // Computes the direction of a melee strike from the given attack position onto the given
    // target cell. CellDirection::UNKNOWN is the canonical "this is a shot" sentinel.
    CellDirection calculateAttackDirection( const Unit & attackingUnit, const Position & attackPosition, const int32_t attackTargetIdx );

    // True if dst denotes the head cell of a position this unit can actually reach on the
    // current turn (and is not the unit's current head cell). Precondition: unit is valid.
    bool isMoveDestinationValid( const Unit * unit, const int32_t dst );

    // Exact gate of Arena::ApplyActionMove: unit exists and is valid, has not acted yet, and
    // dst is a valid move destination.
    bool isMoveCommandValid( const Unit * unit, const int32_t dst );

    // Exact gate of Arena::ApplyActionSkip: unit exists and is valid, and has not acted yet.
    bool isSkipCommandValid( const Unit * unit );

    // Resolved semantics of a valid ATTACK command: the defender cell that will be hit and the
    // strike direction (UNKNOWN for a ranged shot).
    struct ResolvedAttack
    {
        int32_t targetCell{ -1 };
        CellDirection direction{ CellDirection::UNKNOWN };
    };

    // Exact gate of Arena::ApplyActionAttack for the command parameters
    // (attacker, defender, dst move cell or -1, tgt target cell or -1 for auto, dir direction
    // value or negative for auto). Returns the resolved target/direction when the command is
    // legal, std::nullopt when it would be rejected.
    std::optional<ResolvedAttack> resolveAttackCommand( const Unit * attacker, const Unit * defender, const int32_t dst, int32_t tgt, int dir );
}
