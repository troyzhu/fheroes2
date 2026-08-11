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
    record.attack = monster.GetAttack();
    record.defense = monster.GetDefense();
    record.baseStrength = data.battleStats.monsterBaseStrength;
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
        record.reason = "flying movement - needs the flying_v1 profile";
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
    // simple_v1's criteria with the flight exclusion lifted and nothing else: still no wide
    // bodies and no attack shape that changes targeting semantics. A creature that is both
    // wide and flying is admitted by neither profile and waits for a combined one.
    record.flyingV1Supported = record.isValid && !record.isWide && !record.hasTwoCellMeleeAttack && !record.hasAllAdjacentMeleeAttack && !record.hasAreaShot;

    return record;
}

bool fheroes2::agent::isSimpleV1Supported( const int monsterId )
{
    return auditMonster( monsterId ).simpleV1Supported;
}

// Layer-2 semantic adapter. One closed-vocabulary tuple per raw type; every mapping cites the
// engine call site that implements the rule, because the engine stays the authority and this
// table must never drift into folklore. Line numbers are as of this tree; the function names
// beside them survive upstream renumbering.
fheroes2::agent::AbilitySemantics fheroes2::agent::classifyAbility( const MonsterAbilityType type )
{
    switch ( type ) {
    case MonsterAbilityType::DOUBLE_HEX_SIZE:
        // Monster::isWide (monster.h:247-250): the unit occupies two cells; placement, pathing
        // and attack geometry consult the tail cell everywhere.
        return { "always", "self", "movement", "none" };
    case MonsterAbilityType::FLYING:
        // Monster::isFlying (monster.h:242-245), Battle::Unit::isFlying (battle_troop.cpp:331-334):
        // moves over obstacles and units instead of walking.
        return { "always", "self", "movement", "none" };
    case MonsterAbilityType::DRAGON:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:579-582): an attacker under
        // SP_DRAGONSLAYER gains the spell's attack-skill bonus against this unit, so the tag is
        // a defense-time stat modifier on incoming melee and shots.
        return { "on_defense", "self", "stat_mod", "none" };
    case MonsterAbilityType::EARTH_CREATURE:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:568-569, doubling at 576): x2
        // damage when attacking a unit with DOUBLE_DAMAGE_FROM_EARTH_CREATURES.
        return { "on_attack", "enemy_unit", "damage_mult", "none" };
    case MonsterAbilityType::AIR_CREATURE:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:570-571, doubling at 576).
        return { "on_attack", "enemy_unit", "damage_mult", "none" };
    case MonsterAbilityType::FIRE_CREATURE:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:572-573, doubling at 576).
        return { "on_attack", "enemy_unit", "damage_mult", "none" };
    case MonsterAbilityType::WATER_CREATURE:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:574-575, doubling at 576).
        return { "on_attack", "enemy_unit", "damage_mult", "none" };
    case MonsterAbilityType::UNDEAD:
        // getSpellResistance (monster_info.cpp:891-906): 100% resistance to mind-influence and
        // alive-only spells; Monster::isAffectedByMorale (monster.h:287-291): exempt from
        // morale. The Crusader vulnerability (battle_troop.cpp:567) is carried by the
        // attacker's DOUBLE_DAMAGE_TO_UNDEAD record, not duplicated here.
        return { "always", "self", "immunity", "none" };
    case MonsterAbilityType::ELEMENTAL:
        // getSpellResistance (monster_info.cpp:895-897): 100% resistance to mind-influence
        // spells; Monster::isAffectedByMorale (monster.h:287-291): exempt from morale.
        return { "always", "self", "immunity", "none" };
    case MonsterAbilityType::DOUBLE_SHOOTING:
        // Battle::Unit::isDoubleAttack (battle_troop.cpp:1479-1482): a second shot per attack
        // while ammunition lasts, so per-attack damage output doubles.
        return { "on_attack", "enemy_unit", "damage_mult", "none" };
    case MonsterAbilityType::DOUBLE_MELEE_ATTACK:
        // Battle::Unit::isDoubleAttack (battle_troop.cpp:1475-1477): strikes twice in melee.
        return { "on_attack", "enemy_unit", "damage_mult", "none" };
    case MonsterAbilityType::DOUBLE_DAMAGE_TO_UNDEAD:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:567, doubling at 576): x2 damage
        // against UNDEAD defenders.
        return { "on_attack", "enemy_unit", "damage_mult", "none" };
    case MonsterAbilityType::MAGIC_RESISTANCE:
        // getSpellResistance (monster_info.cpp:922-931): `percentage` is the resist chance
        // against damage and hostile spells; 100 means immune to all magic.
        return { "always", "spell_class", "resist", "percent" };
    case MonsterAbilityType::MIND_SPELL_IMMUNITY:
        // getSpellResistance (monster_info.cpp:886-889): 100% against mind-influence spells.
        return { "always", "spell_class", "immunity", "none" };
    case MonsterAbilityType::ELEMENTAL_SPELL_IMMUNITY:
        // getSpellResistance (monster_info.cpp:916-918): 100% against elemental spells.
        return { "always", "spell_class", "immunity", "none" };
    case MonsterAbilityType::FIRE_SPELL_IMMUNITY:
        // getSpellResistance (monster_info.cpp:912-914): 100% against fire spells.
        return { "always", "spell_class", "immunity", "none" };
    case MonsterAbilityType::COLD_SPELL_IMMUNITY:
        // getSpellResistance (monster_info.cpp:908-910): 100% against cold spells.
        return { "always", "spell_class", "immunity", "none" };
    case MonsterAbilityType::IMMUNE_TO_CERTAIN_SPELL:
        // getSpellResistance (monster_info.cpp:933-937): `value` is the Spell id, `percentage`
        // the resist chance against exactly that spell (100 throughout the shipped data).
        return { "always", "spell_class", "resist", "spell_id" };
    case MonsterAbilityType::ELEMENTAL_SPELL_DAMAGE_REDUCTION:
        // Battle::Unit::CalculateSpellDamage (battle_troop.cpp:1302-1305): elemental spell
        // damage is scaled to `percentage` percent.
        return { "always", "spell_class", "resist", "percent" };
    case MonsterAbilityType::CERTAIN_SPELL_DAMAGE_REDUCTION:
        // Battle::Unit::CalculateSpellDamage (battle_troop.cpp:1296-1301): damage from the
        // spell in `value` is scaled to `percentage` percent.
        return { "always", "spell_class", "resist", "spell_id" };
    case MonsterAbilityType::SPELL_CASTER:
        // Battle::Unit::GetSpellMagic (battle_troop.cpp:1536-1552): `percentage` chance per
        // attack to cast the Spell id in `value`; applied to the attacked target after the
        // attack in Battle::Arena::ApplyActionAttack (battle_action.cpp:139-144).
        return { "on_attack", "enemy_unit", "spell_cast", "spell_id" };
    case MonsterAbilityType::HP_REGENERATION:
        // Battle::Unit::NewTurn (battle_troop.cpp:387-391): the injured top creature heals to
        // full hit points at the start of every combat round.
        return { "on_turn", "self", "stat_mod", "none" };
    case MonsterAbilityType::TWO_CELL_MELEE_ATTACK:
        // Battle::Arena::GetTargetsForDamage (battle_action.cpp:725-736): the melee attack also
        // hits the unit in the cell behind the target.
        return { "on_attack", "enemy_unit", "attack_shape", "none" };
    case MonsterAbilityType::UNLIMITED_RETALIATION:
        // Battle::Unit::setRetaliationAsCompleted (battle_troop.cpp:934-941): TR_RETALIATED is
        // never set, so every attack is answered; also retaliates while paralyzed
        // (battle_troop.cpp:461-464).
        return { "on_defense", "self", "retaliation_mod", "none" };
    case MonsterAbilityType::ALL_ADJACENT_CELL_MELEE_ATTACK:
        // Battle::Arena::GetTargetsForDamage (battle_action.cpp:737-751): one melee attack hits
        // every adjacent enemy.
        return { "on_attack", "all_adjacent", "attack_shape", "none" };
    case MonsterAbilityType::NO_MELEE_PENALTY:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:542-544): a shooter without this
        // ability halves its damage in melee; with it, melee damage stays full.
        return { "on_attack", "self", "damage_mult", "none" };
    case MonsterAbilityType::NO_ENEMY_RETALIATION:
        // Monster::isIgnoringRetaliation (monster.h:277-280), consumed in
        // Battle::Arena::ApplyActionAttack (battle_action.cpp:102): the attacked unit never
        // retaliates against this attacker.
        return { "on_attack", "enemy_unit", "retaliation_mod", "none" };
    case MonsterAbilityType::HP_DRAIN:
        // Battle::Unit::ApplyDamage (battle_troop.cpp:778-780): the attacker resurrects its own
        // stack by the hit points of the creatures it killed.
        return { "on_attack", "self", "stat_mod", "none" };
    case MonsterAbilityType::AREA_SHOT:
        // Battle::Arena::GetTargetsForDamage (battle_action.cpp:752-766): a shot also hits
        // every unit adjacent to the target cell, friend or foe.
        return { "on_attack", "all_adjacent", "attack_shape", "none" };
    case MonsterAbilityType::MORAL_DECREMENT:
        // Battle::Unit::GetMorale (battle_troop.cpp:283-286): every morale-affected enemy unit
        // fights at reduced morale while this creature is fielded; `value` is the flat decrement
        // and `percentage` 100 marks it unconditional.
        return { "always", "all_enemies", "stat_mod", "flat" };
    case MonsterAbilityType::ENEMY_HALVING:
        // Battle::Arena::GetTargetsForDamage (battle_action.cpp:699-719): `percentage` chance
        // per attack to replace rolled damage with half the defender stack, when larger.
        return { "on_attack", "enemy_unit", "damage_mult", "percent" };
    case MonsterAbilityType::SOUL_EATER:
        // Battle::Unit::ApplyDamage (battle_troop.cpp:775-777): the attacker grows its own
        // stack by one creature-worth of hit points per creature killed.
        return { "on_attack", "self", "stat_mod", "none" };
    case MonsterAbilityType::NONE:
    default:
        // NONE is a sentinel that populateMonsterData never emits, and an upstream-added type
        // lands here until classified; the regeneration diff of the vendored audit makes either
        // visible.
        return { "always", "self", "other", "none" };
    }
}

fheroes2::agent::AbilitySemantics fheroes2::agent::classifyWeakness( const MonsterWeaknessType type )
{
    switch ( type ) {
    case MonsterWeaknessType::DOUBLE_DAMAGE_FROM_EARTH_CREATURES:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:568-569, doubling at 576): takes
        // x2 melee/shot damage from EARTH_CREATURE attackers.
        return { "on_defense", "self", "damage_mult", "none" };
    case MonsterWeaknessType::DOUBLE_DAMAGE_FROM_AIR_CREATURES:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:570-571, doubling at 576).
        return { "on_defense", "self", "damage_mult", "none" };
    case MonsterWeaknessType::DOUBLE_DAMAGE_FROM_FIRE_CREATURES:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:572-573, doubling at 576).
        return { "on_defense", "self", "damage_mult", "none" };
    case MonsterWeaknessType::DOUBLE_DAMAGE_FROM_WATER_CREATURES:
        // Battle::Unit::CalculateDamageUnit (battle_troop.cpp:574-575, doubling at 576).
        return { "on_defense", "self", "damage_mult", "none" };
    case MonsterWeaknessType::DOUBLE_DAMAGE_FROM_FIRE_SPELLS:
        // Battle::Unit::CalculateSpellDamage (battle_troop.cpp:1315-1318): fire spell damage
        // is doubled.
        return { "always", "spell_class", "damage_mult", "none" };
    case MonsterWeaknessType::DOUBLE_DAMAGE_FROM_COLD_SPELLS:
        // Battle::Unit::CalculateSpellDamage (battle_troop.cpp:1315-1318): cold spell damage
        // is doubled.
        return { "always", "spell_class", "damage_mult", "none" };
    case MonsterWeaknessType::EXTRA_DAMAGE_FROM_CERTAIN_SPELL:
        // Battle::Unit::CalculateSpellDamage (battle_troop.cpp:1309-1314): damage from the
        // spell in `value` is scaled to (100 + `percentage`) percent.
        return { "always", "spell_class", "damage_mult", "spell_id" };
    case MonsterWeaknessType::NONE:
    default:
        // Sentinel / future upstream type; see classifyAbility's default.
        return { "always", "self", "other", "none" };
    }
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
            << ", \"flying_v1_supported\": " << boolText( r.flyingV1Supported ) //
            << ", \"hit_points\": " << r.hitPoints //
            << ", \"attack\": " << r.attack //
            << ", \"defense\": " << r.defense //
            << ", \"base_strength\": " << r.baseStrength //
            << ", \"strength\": " << r.strength;

        // The raw engine ability and weakness records, exported without interpretation: the
        // engine stays the authority on rules, and the learner receives categorical type ids
        // with their typed payloads rather than names (the guide's layer-1 representation;
        // agent_play/docs/research/works/generalized-battle-agent-guide.md). `value` is
        // type-dependent, a spell id for some types and a magnitude for others, so each record
        // additionally carries the layer-2 semantic tuple from classifyAbility/classifyWeakness,
        // appended after the raw fields so layer-1 consumers see identical bytes per field.
        const MonsterData & data = getMonsterData( r.monsterId );
        const auto semanticsText = []( std::ostream & os, const AbilitySemantics & semantics ) {
            os << ", \"trigger\": \"" << semantics.trigger << "\", \"target\": \"" << semantics.target //
               << "\", \"effect\": \"" << semantics.effect << "\", \"magnitude_kind\": \"" << semantics.magnitudeKind << '"';
        };
        out << ", \"abilities\": [";
        for ( size_t k = 0; k < data.battleStats.abilities.size(); ++k ) {
            const MonsterAbility & ability = data.battleStats.abilities[k];
            out << ( k == 0 ? "" : ", " ) << "{\"type_id\": " << static_cast<int>( ability.type ) //
                << ", \"percentage\": " << ability.percentage << ", \"value\": " << ability.value;
            semanticsText( out, classifyAbility( ability.type ) );
            out << '}';
        }
        out << "], \"weaknesses\": [";
        for ( size_t k = 0; k < data.battleStats.weaknesses.size(); ++k ) {
            const MonsterWeakness & weakness = data.battleStats.weaknesses[k];
            out << ( k == 0 ? "" : ", " ) << "{\"type_id\": " << static_cast<int>( weakness.type ) //
                << ", \"percentage\": " << weakness.percentage << ", \"value\": " << weakness.value;
            semanticsText( out, classifyWeakness( weakness.type ) );
            out << '}';
        }
        out << "]" //
            << ", \"reason\": \"" << r.reason << "\"}" //
            << ( i + 1 < records.size() ? ",\n" : "\n" );
    }

    out << "]\n";
    out.flush();

    return out.good();
}
