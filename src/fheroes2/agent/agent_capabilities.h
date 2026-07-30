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

#include <string>
#include <vector>

namespace fheroes2::agent
{
    // Machine-generated capability record for one monster id (agent spec, section 4.2). The
    // audit is derived from the engine's own monster data, never hand-maintained.
    struct MonsterCapability
    {
        int monsterId{ 0 };
        std::string name;
        bool isValid{ false };
        bool isWide{ false };
        bool isFlying{ false };
        bool isShooter{ false };
        bool hasDoubleShooting{ false };
        bool hasDoubleMeleeAttack{ false };
        bool hasTwoCellMeleeAttack{ false };
        bool hasAllAdjacentMeleeAttack{ false };
        bool hasAreaShot{ false };
        bool simpleV1Supported{ false };
        std::string reason;
    };

    MonsterCapability auditMonster( const int monsterId );

    // simple_v1 supports creatures whose ACTION SPACE is plain: single-cell, walking, and
    // without special targeting (no two-cell melee, no all-adjacent melee, no area shot).
    // Abilities that only change outcomes of an ordinary action (double shooting/melee,
    // regeneration, retaliation modifiers, drains...) do not affect the action space and are
    // allowed. Flying is deferred to Phase 1b per spec section 4.4.
    bool isSimpleV1Supported( const int monsterId );

    // Audits every monster id the engine defines (1 .. Monster::MONSTER_COUNT-1).
    std::vector<MonsterCapability> auditAllMonsters();

    // Writes the audit as a JSON array (spec: python/fheroes2_agent/data/monster_capabilities_v1.json).
    // Returns false when the file cannot be opened.
    bool writeCapabilityAudit( const std::string & filePath );
}
