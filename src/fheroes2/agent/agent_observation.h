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

namespace Battle
{
    class Arena;
    class Unit;
}

namespace fheroes2::agent
{
    // What a policy is shown at one decision (agent spec section 12.3, "full_v1" profile of
    // ADR 0001). Read straight off the engine's own Battle::Unit accessors, the same ones the
    // interface calls before drawing a number, so nothing is rendered and nothing is parsed.
    //
    // ADR 0001's rule is that an attribute influencing the transition must be observable or be
    // removed from the dynamics, so speed, shots, morale and luck are all present even though a
    // player reads some of them indirectly.
    struct ObservedUnit
    {
        uint32_t uid{ 0 };
        int monsterId{ 0 };
        bool isAttacker{ false };
        // The stack whose turn it is. Exactly one unit in an observation carries this.
        bool isActive{ false };
        uint32_t count{ 0 };
        uint32_t initialCount{ 0 };
        // Total remaining hit points of the stack, and of its top creature only.
        uint32_t hitPoints{ 0 };
        uint32_t topHitPoints{ 0 };
        uint32_t attack{ 0 };
        uint32_t defense{ 0 };
        uint32_t speed{ 0 };
        uint32_t shots{ 0 };
        int morale{ 0 };
        int luck{ 0 };
        int32_t headCell{ -1 };
        // -1 for single-cell stacks.
        int32_t tailCell{ -1 };
        bool isWide{ false };
        bool isFlying{ false };
        bool isArcher{ false };
        bool isHandFighting{ false };
    };

    struct Observation
    {
        uint32_t engineDecisionIndex{ 0 };
        uint32_t round{ 0 };
        uint32_t activeUid{ 0 };
        bool activeIsAttacker{ false };
        // Living stacks only (agent spec section 12.3), both sides, sorted by uid so the
        // serialization is a function of state alone and not of engine container order.
        std::vector<ObservedUnit> units;
    };

    // Captures the board at the exact pre-application state a decision is taken in. Consumes no
    // combat randomness and mutates nothing, so recording an episode leaves its digests
    // unchanged (asserted by the golden digests in verify_m2.sh and verify_m3.sh).
    Observation captureObservation( const Battle::Arena & arena, const Battle::Unit & currentUnit );

    // One line of JSON, field order fixed, so identical states produce identical bytes.
    std::string observationToJson( const Observation & observation );
}
