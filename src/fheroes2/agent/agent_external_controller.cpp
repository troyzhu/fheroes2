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

#include "agent_external_controller.h"

#include <algorithm>
#include <cassert>

#include "battle_arena.h"
#include "battle_army.h"
#include "battle_troop.h"

bool fheroes2::agent::ExternalDecisionController::handlesDecision( const Battle::Arena & arena, const Battle::Unit & currentUnit ) const
{
    if ( _finished ) {
        // The caller has gone. Handing the decision back to the built-in AI would keep the
        // battle running under a different policy than the one being measured, so the controller
        // keeps the decision and skips it instead (see chooseActions).
        return true;
    }

    if ( _side == ControlledSide::Both ) {
        return true;
    }

    const bool unitIsAttacker = ( currentUnit.GetCurrentOrArmyColor() == arena.getAttackingArmyColor() );
    return ( _side == ControlledSide::Attacker ) ? unitIsAttacker : !unitIsAttacker;
}

void fheroes2::agent::ExternalDecisionController::chooseActions( Battle::Arena & arena, const Battle::Unit & currentUnit, Battle::Actions & output )
{
    ++_seen;

    // One enumeration, exactly as the passive recorder does it, so the legal set an external
    // policy is offered is the same one the coverage audit measured.
    const ActionSet set = enumerateSimpleV1Actions( currentUnit );
    assert( !set.candidates.empty() );

    const auto skip = [&output, &currentUnit]() {
        ActionCandidate fallback;
        fallback.type = CandidateType::Skip;
        fallback.canonicalIndex = actionSkipIndex;
        output.push_back( commandForCandidate( currentUnit, fallback ) );
    };

    if ( _finished || !_decide ) {
        skip();
        return;
    }

    const Observation observation = captureObservation( arena, currentUnit );
    const std::optional<uint32_t> chosen = _decide( observation, set );

    if ( !chosen ) {
        // End of input. Skipping unwinds the arena through its normal path, which is what keeps
        // the terminal state readable rather than leaving a half-applied turn behind.
        _finished = true;
        skip();
        return;
    }

    const auto match = std::find_if( set.candidates.begin(), set.candidates.end(),
                                     [index = *chosen]( const ActionCandidate & candidate ) { return candidate.canonicalIndex == index; } );

    if ( match == set.candidates.end() ) {
        // A selection outside the legal set. Recoverable by contract (agent spec section 5.4):
        // the episode continues with a skip and the caller is told, rather than the worker
        // dying or, worse, the engine being handed an unvalidated command.
        ++_rejected;
        skip();
        return;
    }

    ++_answered;
    output.push_back( commandForCandidate( currentUnit, *match ) );
}
