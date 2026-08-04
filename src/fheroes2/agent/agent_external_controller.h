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
#include <functional>
#include <optional>
#include <string>

#include "agent_action_space.h"
#include "agent_observation.h"
#include "battle_decision_controller.h"

namespace fheroes2::agent
{
    // Which side an external policy plays. The engine's built-in AI keeps the other one, so a
    // battle always has two deciders and the arena never stalls waiting for a second policy.
    enum class ControlledSide : uint8_t
    {
        Attacker,
        Defender,
        Both,
    };

    // Drives one side from outside the engine. The engine owns the call stack, so this does not
    // step: `chooseActions` is called by Arena::UnitTurn and blocks there until the decide
    // callback returns an action, which is the trampoline described in overview.md.
    //
    // The callback receives the board and the legal set and returns a canonical action index.
    // Returning nothing means the caller has gone away, and the controller then falls back to
    // skipping so the episode unwinds cleanly rather than deadlocking.
    class ExternalDecisionController final : public Battle::DecisionController
    {
    public:
        // Returns the chosen canonical index, or nothing to end the episode.
        using DecideFn = std::function<std::optional<uint32_t>( const Observation &, const ActionSet & )>;

        ExternalDecisionController( ControlledSide side, DecideFn decide )
            : _side( side )
            , _decide( std::move( decide ) )
        {
            // Intentionally empty.
        }

        bool handlesDecision( const Battle::Arena & arena, const Battle::Unit & currentUnit ) const override;

        void chooseActions( Battle::Arena & arena, const Battle::Unit & currentUnit, Battle::Actions & output ) override;

        // Number of decisions this controller was asked for, and how many it answered. They
        // differ once the caller has closed the connection.
        uint32_t decisionsSeen() const
        {
            return _seen;
        }

        uint32_t decisionsAnswered() const
        {
            return _answered;
        }

        // Set once the callback declines, so a driver can stop rather than run the battle out.
        bool isFinished() const
        {
            return _finished;
        }

        // A selection outside the legal set, which is a caller error rather than an engine one.
        uint32_t rejectedSelections() const
        {
            return _rejected;
        }

    private:
        ControlledSide _side;
        DecideFn _decide;
        uint32_t _seen{ 0 };
        uint32_t _answered{ 0 };
        uint32_t _rejected{ 0 };
        bool _finished{ false };
    };
}
