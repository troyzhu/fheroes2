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

// The function bodies below were moved verbatim from the anonymous namespace and the
// checkParameters lambdas of battle_action.cpp so that Arena::ApplyAction{Move,Attack,Skip}
// and external candidate enumeration validate through one implementation. Any change here is
// a change to battle legality itself: keep them in lockstep with the ApplyAction effects and
// treat digest regressions in the agent verification suites as authoritative.

#include "battle_action_validation.h"

#include <cassert>

#include "battle.h"
#include "battle_board.h"
#include "battle_troop.h"

int32_t Battle::calculateAttackTarget( const Unit & attackingUnit, const Position & attackPosition, const Unit & defendingUnit )
{
    const int32_t attackPositionHeadIdx = attackPosition.GetHead() ? attackPosition.GetHead()->GetIndex() : -1;
    const int32_t attackPositionTailIdx = attackPosition.GetTail() ? attackPosition.GetTail()->GetIndex() : -1;

    assert( attackPositionHeadIdx != -1 && ( attackingUnit.isWide() ? attackPositionTailIdx != -1 : attackPositionTailIdx == -1 ) );

    if ( Board::CanAttackFromCell( attackingUnit, attackPositionHeadIdx ) ) {
        // The defender's head cell is near the head cell of the attack position
        if ( Board::isNearIndexes( attackPositionHeadIdx, defendingUnit.GetHeadIndex() ) ) {
            return defendingUnit.GetHeadIndex();
        }

        // The defender's tail cell is near the head cell of the attack position
        if ( defendingUnit.isWide() && Board::isNearIndexes( attackPositionHeadIdx, defendingUnit.GetTailIndex() ) ) {
            return defendingUnit.GetTailIndex();
        }
    }

    if ( Board::CanAttackFromCell( attackingUnit, attackPositionTailIdx ) ) {
        // The defender's head cell is near the tail cell of the attack position
        if ( Board::isNearIndexes( attackPositionTailIdx, defendingUnit.GetHeadIndex() ) ) {
            return defendingUnit.GetHeadIndex();
        }

        // The defender's tail cell is near the tail cell of the attack position
        if ( defendingUnit.isWide() && Board::isNearIndexes( attackPositionTailIdx, defendingUnit.GetTailIndex() ) ) {
            return defendingUnit.GetTailIndex();
        }
    }

    // Attack position is not near the defender, this is most likely a shot
    return defendingUnit.GetHeadIndex();
}

Battle::CellDirection Battle::calculateAttackDirection( const Unit & attackingUnit, const Position & attackPosition, const int32_t attackTargetIdx )
{
    const int32_t attackPositionHeadIdx = attackPosition.GetHead() ? attackPosition.GetHead()->GetIndex() : -1;
    const int32_t attackPositionTailIdx = attackPosition.GetTail() ? attackPosition.GetTail()->GetIndex() : -1;

    assert( attackPositionHeadIdx != -1 && ( attackingUnit.isWide() ? attackPositionTailIdx != -1 : attackPositionTailIdx == -1 ) );

    // The target cell of the attack is near the head cell of the attack position
    if ( Board::CanAttackFromCell( attackingUnit, attackPositionHeadIdx ) && Board::isNearIndexes( attackPositionHeadIdx, attackTargetIdx ) ) {
        return Board::GetDirection( attackPositionHeadIdx, attackTargetIdx );
    }

    // The target cell of the attack is near the tail cell of the attack position
    if ( Board::CanAttackFromCell( attackingUnit, attackPositionTailIdx ) && Board::isNearIndexes( attackPositionTailIdx, attackTargetIdx ) ) {
        return Board::GetDirection( attackPositionTailIdx, attackTargetIdx );
    }

    // Attack position is not near the defender, this is most likely a shot
    return CellDirection::UNKNOWN;
}

bool Battle::isMoveDestinationValid( const Unit * unit, const int32_t dst )
{
    assert( unit != nullptr && unit->isValid() );

    // "Moving" a unit to its current position is not allowed
    if ( unit->GetHeadIndex() == dst ) {
        return false;
    }

    const Position pos = Position::GetReachable( *unit, dst );
    if ( pos.GetHead() == nullptr ) {
        return false;
    }

    assert( pos.isValidForUnit( unit ) );

    // Index of the destination cell should correspond to the index of the head cell of the target position and nothing else
    return pos.GetHead()->GetIndex() == dst;
}

bool Battle::isMoveCommandValid( const Unit * unit, const int32_t dst )
{
    if ( unit == nullptr || !unit->isValid() ) {
        return false;
    }

    if ( unit->Modes( TR_MOVED ) ) {
        return false;
    }

    if ( !isMoveDestinationValid( unit, dst ) ) {
        return false;
    }

    return true;
}

bool Battle::isSkipCommandValid( const Unit * unit )
{
    if ( unit == nullptr || !unit->isValid() ) {
        return false;
    }

    if ( unit->Modes( TR_MOVED ) ) {
        return false;
    }

    return true;
}

std::optional<Battle::ResolvedAttack> Battle::resolveAttackCommand( const Unit * attacker, const Unit * defender, const int32_t dst, int32_t tgt, int dir )
{
    if ( attacker == nullptr || !attacker->isValid() ) {
        return std::nullopt;
    }

    if ( defender == nullptr || !defender->isValid() ) {
        return std::nullopt;
    }

    if ( attacker->Modes( TR_MOVED ) ) {
        return std::nullopt;
    }

    if ( attacker->GetCurrentColor() == defender->GetColor() ) {
        return std::nullopt;
    }

    // Attacker can attack from his current position without performing a move (in this case, the index of the destination cell should be -1)
    if ( dst != -1 && !isMoveDestinationValid( attacker, dst ) ) {
        return std::nullopt;
    }

    if ( attacker->isArchers() && !attacker->isHandFighting() ) {
        // Non-blocked archer can only attack by shooting from his current position
        if ( dst != -1 ) {
            return std::nullopt;
        }

        if ( tgt < 0 ) {
            tgt = calculateAttackTarget( *attacker, attacker->GetPosition(), *defender );
        }

        const CellDirection cellDir = dir < 0 ? calculateAttackDirection( *attacker, attacker->GetPosition(), tgt ) : static_cast<CellDirection>( dir );

        if ( !defender->GetPosition().contains( tgt ) ) {
            return std::nullopt;
        }

        // Non-blocked archers cannot attack "from a direction"
        if ( cellDir != CellDirection::UNKNOWN ) {
            return std::nullopt;
        }

        return ResolvedAttack{ tgt, cellDir };
    }

    const Position attackPos = ( dst == -1 ? attacker->GetPosition() : Position::GetReachable( *attacker, dst ) );
    if ( attackPos.GetHead() == nullptr ) {
        return std::nullopt;
    }

    assert( attackPos.isValidForUnit( attacker ) );

    if ( tgt < 0 ) {
        tgt = calculateAttackTarget( *attacker, attackPos, *defender );
    }

    const CellDirection cellDir = dir < 0 ? calculateAttackDirection( *attacker, attackPos, tgt ) : static_cast<CellDirection>( dir );

    if ( !defender->GetPosition().contains( tgt ) ) {
        return std::nullopt;
    }

    // Melee attacks are only possible from a certain direction
    if ( cellDir == CellDirection::UNKNOWN ) {
        return std::nullopt;
    }

    const CellDirection reflectDir = Board::GetReflectDirection( cellDir );
    const int32_t attackIdx = ( Board::isValidDirection( tgt, reflectDir ) ? Board::GetIndexDirection( tgt, reflectDir ) : -1 );

    if ( !attackPos.contains( attackIdx ) ) {
        return std::nullopt;
    }

    // Attack from a specified cell may be prohibited - for example, if this cell belongs to a castle moat
    if ( !Board::CanAttackFromCell( *attacker, attackIdx ) ) {
        return std::nullopt;
    }

    return ResolvedAttack{ tgt, cellDir };
}
