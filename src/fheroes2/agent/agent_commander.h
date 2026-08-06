/***************************************************************************
 *   Free Heroes of Might and Magic II: https://github.com/ihhub/fheroes2  *
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

#include "army.h"
#include "heroes_base.h"
#include "luck.h"
#include "morale.h"
#include "players.h"
#include "race.h"

class Castle;

namespace fheroes2::agent
{
    // A minimal commander that exists only to carry primary stats into a scenario battle.
    //
    // Every stat a battle unit reads flows through ArmyTroop::GetAttack / GetDefense, which add
    // the army commander's attack and defense when a commander is present. Real maps always have
    // one, so a commander-less scenario understates every unit on both sides, and the Thunk
    // opening fight showed the gap is decisive. This class is the smallest object that closes it:
    // stats and nothing else.
    //
    // The deliberately inert choices, each load-bearing:
    //   - GetType() is CAPTAIN, because the engine forbids captains to retreat or surrender,
    //     which removes the battle-flow branches a real hero would open (those need a kingdom
    //     and treasury this headless world does not fully populate).
    //   - GetControl() is CONTROL_AI by default, because Army::GetControl defers to the commander
    //     when one is set, and a human-controlled army would stall the headless loop waiting for
    //     input. The interactive replay tool passes CONTROL_HUMAN for the side a person plays,
    //     which is the only way the battle interface ever takes that army's turns; every headless
    //     caller keeps the default and is unaffected.
    //   - No spell book is ever added, so CanCastSpell is false and the AI's casting logic
    //     skips the commander entirely; spell power is the engine's floor value and unused.
    class ScenarioCommander final : public HeroBase
    {
    public:
        ScenarioCommander( Army & army, const PlayerColor color, const int attackStat, const int defenseStat,
                           const int control = CONTROL_AI )
            : _army( army )
            , _color( color )
            , _control( control )
        {
            attack = attackStat;
            defense = defenseStat;
            power = 1;
            knowledge = 0;
        }

        ScenarioCommander( const ScenarioCommander & ) = delete;
        ScenarioCommander & operator=( const ScenarioCommander & ) = delete;

        const std::string & GetName() const override
        {
            static const std::string name( "Scenario Commander" );
            return name;
        }

        PlayerColor GetColor() const override
        {
            return _color;
        }

        int GetControl() const override
        {
            return _control;
        }

        bool isValid() const override
        {
            return true;
        }

        const Army & GetArmy() const override
        {
            return _army;
        }

        Army & GetArmy() override
        {
            return _army;
        }

        uint32_t GetMaxSpellPoints() const override
        {
            return 0;
        }

        int GetLevelSkill( const int /* skill */ ) const override
        {
            return Skill::Level::NONE;
        }

        uint32_t GetSecondarySkillValue( const int /* skill */ ) const override
        {
            return 0;
        }

        void ActionAfterBattle() override
        {
            // Nothing to do: no experience, no spell point regeneration, no map state.
        }

        void ActionPreBattle() override
        {
            // Nothing to do.
        }

        const Castle * inCastle() const override
        {
            return nullptr;
        }

        void PortraitRedraw( const int32_t /* px */, const int32_t /* py */, const PortraitType /* type */, fheroes2::Image & /* dstsf */ ) const override
        {
            // Headless: there is no surface to draw a portrait onto.
        }

        int GetType() const override
        {
            return HeroBase::CAPTAIN;
        }

        int GetAttack() const override
        {
            return attack;
        }

        int GetDefense() const override
        {
            return defense;
        }

        int GetPower() const override
        {
            return power;
        }

        int GetKnowledge() const override
        {
            return knowledge;
        }

        int GetMorale() const override
        {
            return Morale::NORMAL;
        }

        int GetLuck() const override
        {
            return Luck::NORMAL;
        }

        int GetRace() const override
        {
            return Race::NONE;
        }

    private:
        Army & _army;
        PlayerColor _color;
        int _control{ CONTROL_AI };
    };
}
