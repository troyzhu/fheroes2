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

#include "agent_capabilities.h"

#include <fstream>

#include "monster.h"
#include "monster_info.h"

fheroes2::agent::MonsterCapability fheroes2::agent::auditMonster( const int monsterId )
{
    MonsterCapability record;
    record.monsterId = monsterId;

    const Monster monster( monsterId );
    record.name = monster.GetName();
    record.isValid = monster.isValid();

    if ( !record.isValid ) {
        record.reason = "not a concrete battle creature";
        return record;
    }

    const MonsterData & data = getMonsterData( monsterId );
    const std::vector<MonsterAbility> & abilities = data.battleStats.abilities;

    record.hitPoints = monster.GetHitPoints();
    record.strength = monster.GetMonsterStrength();
    record.isWide = monster.isWide();
    record.isFlying = monster.isFlying();
    record.isShooter = ( data.battleStats.shots > 0 );
    record.hasDoubleShooting = isAbilityPresent( abilities, MonsterAbilityType::DOUBLE_SHOOTING );
    record.hasDoubleMeleeAttack = isAbilityPresent( abilities, MonsterAbilityType::DOUBLE_MELEE_ATTACK );
    record.hasTwoCellMeleeAttack = isAbilityPresent( abilities, MonsterAbilityType::TWO_CELL_MELEE_ATTACK );
    record.hasAllAdjacentMeleeAttack = isAbilityPresent( abilities, MonsterAbilityType::ALL_ADJACENT_CELL_MELEE_ATTACK );
    record.hasAreaShot = isAbilityPresent( abilities, MonsterAbilityType::AREA_SHOT );

    // Exclusions are action-space based; the first matching reason is recorded.
    if ( record.isWide ) {
        record.reason = "wide (two-cell) unit - deferred to Phase 1b";
    }
    else if ( record.isFlying ) {
        record.reason = "flying movement - deferred to Phase 1b";
    }
    else if ( record.hasTwoCellMeleeAttack ) {
        record.reason = "two-cell melee attack changes targeting semantics";
    }
    else if ( record.hasAllAdjacentMeleeAttack ) {
        record.reason = "all-adjacent melee attack changes targeting semantics";
    }
    else if ( record.hasAreaShot ) {
        record.reason = "area shot changes ranged targeting semantics";
    }
    else {
        record.simpleV1Supported = true;
        record.reason = record.isShooter ? "single-cell walking shooter" : "single-cell walking melee";
    }

    record.wideV1Supported = record.isValid && !record.isFlying && !record.hasTwoCellMeleeAttack && !record.hasAllAdjacentMeleeAttack && !record.hasAreaShot;

    return record;
}

bool fheroes2::agent::isSimpleV1Supported( const int monsterId )
{
    return auditMonster( monsterId ).simpleV1Supported;
}

std::vector<fheroes2::agent::MonsterCapability> fheroes2::agent::auditAllMonsters()
{
    std::vector<MonsterCapability> records;
    records.reserve( Monster::MONSTER_COUNT - 1 );

    for ( int id = Monster::UNKNOWN + 1; id < Monster::MONSTER_COUNT; ++id ) {
        records.push_back( auditMonster( id ) );
    }

    return records;
}

bool fheroes2::agent::writeCapabilityAudit( const std::string & filePath )
{
    std::ofstream out( filePath, std::ios_base::out | std::ios_base::trunc );
    if ( !out.is_open() ) {
        return false;
    }

    const auto boolText = []( const bool value ) { return value ? "true" : "false"; };

    out << "[\n";

    const std::vector<MonsterCapability> records = auditAllMonsters();
    for ( size_t i = 0; i < records.size(); ++i ) {
        const MonsterCapability & r = records[i];

        // Names come from the engine's own monster table; none contain characters needing
        // JSON escaping beyond what plain ASCII provides.
        out << "  {\"monster_id\": " << r.monsterId //
            << ", \"name\": \"" << r.name << '"' //
            << ", \"is_valid\": " << boolText( r.isValid ) //
            << ", \"is_wide\": " << boolText( r.isWide ) //
            << ", \"is_flying\": " << boolText( r.isFlying ) //
            << ", \"is_archer\": " << boolText( r.isShooter ) //
            << ", \"has_double_shooting\": " << boolText( r.hasDoubleShooting ) //
            << ", \"has_double_melee_attack\": " << boolText( r.hasDoubleMeleeAttack ) //
            << ", \"is_double_cell_attack\": " << boolText( r.hasTwoCellMeleeAttack ) //
            << ", \"has_area_or_multi_target_attack\": " << boolText( r.hasAllAdjacentMeleeAttack || r.hasAreaShot ) //
            << ", \"simple_v1_supported\": " << boolText( r.simpleV1Supported ) //
            << ", \"wide_v1_supported\": " << boolText( r.wideV1Supported ) //
            << ", \"hit_points\": " << r.hitPoints //
            << ", \"strength\": " << r.strength //
            << ", \"reason\": \"" << r.reason << "\"}" //
            << ( i + 1 < records.size() ? ",\n" : "\n" );
    }

    out << "]\n";
    out.flush();

    return out.good();
}
