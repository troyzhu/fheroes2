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

namespace Battle
{
    class Actions;
    class Arena;
    class Unit;

    // Optional hook into the full-fledged unit-decision branch of Arena::UnitTurn. A null
    // controller (the default for every existing caller) leaves engine behavior completely
    // unchanged.
    //
    // The hook is consulted only where the engine itself would ask the built-in AI or the human
    // interface for a decision. It is never consulted for pending UI actions, units unable to
    // act, or automatic bad-morale actions, and implementations must not apply commands, consume
    // combat randomness, or mutate the arena while observing.
    class DecisionController
    {
    public:
        DecisionController() = default;
        DecisionController( const DecisionController & ) = delete;
        DecisionController & operator=( const DecisionController & ) = delete;

        virtual ~DecisionController() = default;

        // Returns true if this controller wants to choose the actions for the given unit's
        // decision instead of the built-in AI or the human interface.
        virtual bool handlesDecision( const Arena & arena, const Unit & currentUnit ) const = 0;

        // Called only when handlesDecision() returned true for this decision. Must append at
        // least one valid action to the output.
        virtual void chooseActions( Arena & arena, const Unit & currentUnit, Actions & output ) = 0;

        // Called for every full-fledged decision - whether the actions were chosen by this
        // controller, the built-in AI, or a human - after the actions have been chosen but
        // before the command stream updates the combat random generator and before any command
        // is applied.
        virtual void observeChosenActions( const Arena & /* arena */, const Unit & /* currentUnit */, const Actions & /* actions */ )
        {
            // Intentionally empty: observing is optional.
        }
    };
}
