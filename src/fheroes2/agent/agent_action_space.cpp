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

#include "agent_action_space.h"

#include <algorithm>
#include <array>
#include <cassert>

#include "battle_action_validation.h"
#include "battle_arena.h"
#include "battle_army.h"
#include "battle_board.h"
#include "battle_troop.h"

namespace
{
    using fheroes2::agent::ActionCandidate;
    using fheroes2::agent::ActionSet;
    using fheroes2::agent::CandidateType;

    void addCandidate( ActionSet & set, ActionCandidate candidate )
    {
        assert( candidate.canonicalIndex < fheroes2::agent::actionSpaceSize );

        if ( set.legalMask[candidate.canonicalIndex] != 0 ) {
            // The same canonical action can be probed through several construction paths
            // (e.g. a defender cell adjacent to two of the attacker's reachable cells);
            // keep the first.
            return;
        }

        set.legalMask[candidate.canonicalIndex] = 1;
        set.candidates.push_back( std::move( candidate ) );
    }

    std::string makeKey( const ActionCandidate & c )
    {
        std::string key = fheroes2::agent::candidateTypeName( c.type );
        switch ( c.type ) {
        case CandidateType::Skip:
            break;
        case CandidateType::Move:
            key += ':';
            key += std::to_string( c.moveCell );
            break;
        case CandidateType::RangedAttack:
            key += ':';
            key += std::to_string( c.defenderUid );
            key += ':';
            key += std::to_string( c.resolvedTargetCell );
            break;
        case CandidateType::MeleeAttack:
            key += ':';
            key += std::to_string( c.defenderUid );
            key += ':';
            key += std::to_string( c.moveCell );
            key += ':';
            key += std::to_string( c.resolvedTargetCell );
            key += ':';
            key += std::to_string( static_cast<int>( c.resolvedDirection ) );
            break;
        default:
            assert( 0 );
            break;
        }
        return key;
    }

    void enumerateAttacksOnEnemy( ActionSet & set, const Battle::Unit & activeUnit, const Battle::Unit & enemy, const bool canShoot )
    {
        if ( canShoot ) {
            // One canonical ranged action per living enemy stack; the command uses engine
            // auto-resolution (tgt = -1) and direction 0, exactly like the built-in AI's shots.
            const std::optional<Battle::ResolvedAttack> resolved = Battle::resolveAttackCommand( &activeUnit, &enemy, -1, -1, 0 );
            if ( resolved ) {
                assert( resolved->direction == Battle::CellDirection::UNKNOWN );

                ActionCandidate candidate;
                candidate.canonicalIndex = fheroes2::agent::actionRangedBase + static_cast<uint32_t>( enemy.GetHeadIndex() );
                candidate.type = CandidateType::RangedAttack;
                candidate.defenderUid = enemy.GetUID();
                candidate.moveCell = -1;
                candidate.targetCell = -1;
                candidate.direction = 0;
                candidate.resolvedTargetCell = resolved->targetCell;
                candidate.resolvedDirection = resolved->direction;
                candidate.canonicalKey = makeKey( candidate );
                addCandidate( set, std::move( candidate ) );
            }

            return;
        }

        // Melee: for every cell of the enemy and every strike direction, derive the single
        // attacker cell that realizes (target, direction) - the exact inverse the engine's
        // validator performs - and let the shared resolver decide legality.
        const std::array<int32_t, 2> enemyCells = { enemy.GetHeadIndex(), enemy.isWide() ? enemy.GetTailIndex() : -1 };

        for ( const int32_t targetCell : enemyCells ) {
            if ( targetCell < 0 ) {
                continue;
            }

            for ( Battle::CellDirection dir = Battle::CellDirection::TOP_LEFT; dir < Battle::CellDirection::CENTER; ++dir ) {
                const Battle::CellDirection reflectDir = Battle::Board::GetReflectDirection( dir );
                if ( !Battle::Board::isValidDirection( targetCell, reflectDir ) ) {
                    continue;
                }

                const int32_t attackFromCell = Battle::Board::GetIndexDirection( targetCell, reflectDir );
                const int32_t dst = ( attackFromCell == activeUnit.GetHeadIndex() ) ? -1 : attackFromCell;

                const std::optional<Battle::ResolvedAttack> resolved = Battle::resolveAttackCommand( &activeUnit, &enemy, dst, targetCell, static_cast<int>( dir ) );
                if ( !resolved ) {
                    continue;
                }

                assert( resolved->targetCell == targetCell && resolved->direction == dir );

                const int dirIdx = fheroes2::agent::meleeDirectionIndex( dir );
                assert( dirIdx >= 0 );

                ActionCandidate candidate;
                candidate.canonicalIndex
                    = fheroes2::agent::actionMeleeBase + static_cast<uint32_t>( targetCell ) * 6 + static_cast<uint32_t>( dirIdx );
                candidate.type = CandidateType::MeleeAttack;
                candidate.defenderUid = enemy.GetUID();
                candidate.moveCell = dst;
                candidate.targetCell = targetCell;
                candidate.direction = static_cast<int32_t>( dir );
                candidate.resolvedTargetCell = resolved->targetCell;
                candidate.resolvedDirection = resolved->direction;
                candidate.canonicalKey = makeKey( candidate );
                addCandidate( set, std::move( candidate ) );
            }
        }
    }
}

int fheroes2::agent::meleeDirectionIndex( const Battle::CellDirection direction )
{
    switch ( direction ) {
    case Battle::CellDirection::TOP_LEFT:
        return 0;
    case Battle::CellDirection::TOP_RIGHT:
        return 1;
    case Battle::CellDirection::RIGHT:
        return 2;
    case Battle::CellDirection::BOTTOM_RIGHT:
        return 3;
    case Battle::CellDirection::BOTTOM_LEFT:
        return 4;
    case Battle::CellDirection::LEFT:
        return 5;
    default:
        return -1;
    }
}

Battle::CellDirection fheroes2::agent::meleeDirectionFromIndex( const int index )
{
    switch ( index ) {
    case 0:
        return Battle::CellDirection::TOP_LEFT;
    case 1:
        return Battle::CellDirection::TOP_RIGHT;
    case 2:
        return Battle::CellDirection::RIGHT;
    case 3:
        return Battle::CellDirection::BOTTOM_RIGHT;
    case 4:
        return Battle::CellDirection::BOTTOM_LEFT;
    case 5:
        return Battle::CellDirection::LEFT;
    default:
        return Battle::CellDirection::UNKNOWN;
    }
}

const char * fheroes2::agent::candidateTypeName( const CandidateType type )
{
    switch ( type ) {
    case CandidateType::Skip:
        return "skip";
    case CandidateType::Move:
        return "move";
    case CandidateType::RangedAttack:
        return "ranged";
    case CandidateType::MeleeAttack:
        return "melee";
    default:
        assert( 0 );
        return "unknown";
    }
}

fheroes2::agent::ActionSet fheroes2::agent::enumerateSimpleV1Actions( const Battle::Unit & activeUnit )
{
    ActionSet set;
    set.legalMask.assign( actionSpaceSize, 0 );

    Battle::Arena * arena = Battle::GetArena();
    assert( arena != nullptr );

    // SKIP is validated through the same gate the engine applies it with.
    if ( Battle::isSkipCommandValid( &activeUnit ) ) {
        ActionCandidate candidate;
        candidate.canonicalIndex = actionSkipIndex;
        candidate.type = CandidateType::Skip;
        candidate.canonicalKey = makeKey( candidate );
        addCandidate( set, std::move( candidate ) );
    }

    // Moves: the pathfinder yields reachable head cells; each one is confirmed through the
    // exact ApplyActionMove gate.
    for ( const int32_t dst : arena->getAllAvailableMoves( activeUnit ) ) {
        if ( !Battle::isMoveCommandValid( &activeUnit, dst ) ) {
            continue;
        }

        ActionCandidate candidate;
        candidate.canonicalIndex = actionMoveBase + static_cast<uint32_t>( dst );
        candidate.type = CandidateType::Move;
        candidate.moveCell = dst;
        candidate.canonicalKey = makeKey( candidate );
        addCandidate( set, std::move( candidate ) );
    }

    // Attacks: ranged when the engine's shooting rule says so, melee otherwise - the same
    // isArchers/isHandFighting branch the validator itself applies.
    const bool canShoot = activeUnit.isArchers() && !activeUnit.isHandFighting();

    const auto enumerateForce = [&]( const Battle::Force & force ) {
        for ( const Battle::Unit * enemy : force ) {
            if ( enemy == nullptr || !enemy->isValid() ) {
                continue;
            }

            if ( activeUnit.GetCurrentColor() == enemy->GetColor() ) {
                continue;
            }

            enumerateAttacksOnEnemy( set, activeUnit, *enemy, canShoot );
        }
    };

    enumerateForce( arena->getAttackingForce() );
    enumerateForce( arena->getDefendingForce() );

    std::sort( set.candidates.begin(), set.candidates.end(),
               []( const ActionCandidate & a, const ActionCandidate & b ) { return a.canonicalIndex < b.canonicalIndex; } );

    return set;
}

std::optional<uint32_t> fheroes2::agent::resolveTeacherActionIndex( const Battle::Unit & activeUnit, const std::vector<CommandSnapshot> & actions )
{
    if ( actions.size() != 1 ) {
        return std::nullopt;
    }

    const CommandSnapshot & action = actions[0];

    switch ( action.type ) {
    case Battle::CommandType::SKIP:
        if ( action.unitUid != activeUnit.GetUID() ) {
            return std::nullopt;
        }
        return actionSkipIndex;

    case Battle::CommandType::MOVE:
        if ( action.unitUid != activeUnit.GetUID() || action.moveCell < 0 || action.moveCell >= 99 ) {
            return std::nullopt;
        }
        return actionMoveBase + static_cast<uint32_t>( action.moveCell );

    case Battle::CommandType::ATTACK: {
        if ( action.unitUid != activeUnit.GetUID() ) {
            return std::nullopt;
        }

        const Battle::Arena * arena = Battle::GetArena();
        assert( arena != nullptr );

        const Battle::Unit * defender = arena->GetTroopUID( action.defenderUid );
        if ( defender == nullptr ) {
            return std::nullopt;
        }

        const std::optional<Battle::ResolvedAttack> resolved
            = Battle::resolveAttackCommand( &activeUnit, defender, action.moveCell, action.targetCell, action.direction );
        if ( !resolved ) {
            return std::nullopt;
        }

        if ( resolved->direction == Battle::CellDirection::UNKNOWN ) {
            return actionRangedBase + static_cast<uint32_t>( defender->GetHeadIndex() );
        }

        const int dirIdx = meleeDirectionIndex( resolved->direction );
        if ( dirIdx < 0 || resolved->targetCell < 0 || resolved->targetCell >= 99 ) {
            return std::nullopt;
        }

        return actionMeleeBase + static_cast<uint32_t>( resolved->targetCell ) * 6 + static_cast<uint32_t>( dirIdx );
    }

    default:
        return std::nullopt;
    }
}
