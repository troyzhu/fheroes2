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
#include <string>
#include <vector>

#include "agent_command_snapshot.h"
#include "battle_cell.h"

namespace Battle
{
    class Unit;
}

namespace fheroes2::agent
{
    // Canonical fixed action indexing for the simple_v1 profile over the 11x9 battlefield
    // (ADR 0002, "actions_simple_v1" schema). The index of an action is a pure function of
    // board geometry and the action taxonomy - stable across states and episodes:
    //
    //   [0]                     SKIP
    //   [1   .. 99]             MOVE to head cell c            (1 + c)
    //   [100 .. 198]            RANGED attack on the enemy whose head cell is c   (100 + c)
    //   [199 .. 792]            MELEE strike onto target cell t from direction d  (199 + t*6 + d')
    //
    // where d' enumerates the six real hex directions in enum order (TOP_LEFT..LEFT). The
    // boolean legal mask and the candidate list below are two views derived from ONE
    // enumeration that validates through battle_action_validation - the same gates
    // Arena::ApplyAction* executes with.
    inline constexpr const char * actionSchemaName = "actions_simple_v1";

    inline constexpr uint32_t actionSkipIndex{ 0 };
    inline constexpr uint32_t actionMoveBase{ 1 };
    inline constexpr uint32_t actionRangedBase{ 1 + 99 };
    inline constexpr uint32_t actionMeleeBase{ 1 + 99 + 99 };
    inline constexpr uint32_t actionSpaceSize{ 1 + 99 + 99 + 99 * 6 };

    // 0..5 for the six real hex directions (TOP_LEFT, TOP_RIGHT, RIGHT, BOTTOM_RIGHT,
    // BOTTOM_LEFT, LEFT); -1 for UNKNOWN/CENTER.
    int meleeDirectionIndex( const Battle::CellDirection direction );
    Battle::CellDirection meleeDirectionFromIndex( const int index );

    enum class CandidateType : uint8_t
    {
        Skip,
        Move,
        RangedAttack,
        MeleeAttack,
    };

    const char * candidateTypeName( const CandidateType type );

    // One legal action: the canonical index plus the exact engine command parameters that
    // realize it (spec 10.1 - the external policy only ever selects an index; the command is
    // engine-owned).
    struct ActionCandidate
    {
        uint32_t canonicalIndex{ 0 };
        CandidateType type{ CandidateType::Skip };
        uint32_t defenderUid{ 0 }; // attacks only
        int32_t moveCell{ -1 };    // ATTACK/MOVE dst parameter; -1 = act from current position
        int32_t targetCell{ -1 };  // ATTACK tgt parameter as sent (-1 = engine auto-resolution)
        int32_t direction{ 0 };    // ATTACK dir parameter as sent
        int32_t resolvedTargetCell{ -1 };
        Battle::CellDirection resolvedDirection{ Battle::CellDirection::UNKNOWN };
        std::string canonicalKey;
    };

    struct ActionSet
    {
        // actionSpaceSize entries, 0/1. Invariant: mask[i] == 1 exactly when a candidate with
        // canonicalIndex i exists.
        std::vector<uint8_t> legalMask;
        // Sorted by canonicalIndex (deterministic ordering, spec 10.4).
        std::vector<ActionCandidate> candidates;
    };

    // Enumerates the legal simple_v1 actions of the active unit against the process-global
    // arena, at the current point of the battle (call it from a DecisionController before any
    // command of the decision is applied). Non-mutating apart from warming the pathfinder
    // cache; consumes no combat randomness.
    ActionSet enumerateSimpleV1Actions( const Battle::Unit & activeUnit );

    // Maps an observed teacher decision (its command snapshots, taken before application) onto
    // the canonical action index, using the same resolver as enumeration. std::nullopt when
    // the decision is outside simple_v1 (multi-command decisions, morale, spells, ...).
    std::optional<uint32_t> resolveTeacherActionIndex( const Battle::Unit & activeUnit, const std::vector<CommandSnapshot> & actions );
}
