/***************************************************************************
 *   fheroes2 agent tests: monster capability audit and simple_v1 allowlist *
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>

#include <unistd.h>

#include "agent_capabilities.h"
#include "monster.h"
#include "monster_info.h"
#include "spell.h"

namespace
{
    int passed = 0;
    int failed = 0;

    void check( const bool condition, const char * name )
    {
        if ( condition ) {
            std::printf( "  PASS  %s\n", name );
            ++passed;
        }
        else {
            std::printf( "  FAIL  %s\n", name );
            ++failed;
        }
    }

    bool oneOf( const char * candidate, const std::initializer_list<const char *> & vocabulary )
    {
        if ( candidate == nullptr ) {
            return false;
        }
        for ( const char * word : vocabulary ) {
            if ( std::strcmp( candidate, word ) == 0 ) {
                return true;
            }
        }
        return false;
    }

    bool semanticsEqual( const fheroes2::agent::AbilitySemantics & semantics, const char * trigger, const char * target, const char * effect, const char * magnitude )
    {
        return semantics.trigger != nullptr && semantics.target != nullptr && semantics.effect != nullptr && semantics.magnitudeKind != nullptr
               && std::strcmp( semantics.trigger, trigger ) == 0 && std::strcmp( semantics.target, target ) == 0 && std::strcmp( semantics.effect, effect ) == 0
               && std::strcmp( semantics.magnitudeKind, magnitude ) == 0;
    }

    const fheroes2::MonsterAbility * findAbility( const int monsterId, const fheroes2::MonsterAbilityType type )
    {
        for ( const fheroes2::MonsterAbility & ability : fheroes2::getMonsterData( monsterId ).battleStats.abilities ) {
            if ( ability.type == type ) {
                return &ability;
            }
        }
        return nullptr;
    }

    size_t countOccurrences( const std::string & haystack, const std::string & needle )
    {
        size_t count = 0;
        for ( size_t pos = haystack.find( needle ); pos != std::string::npos; pos = haystack.find( needle, pos + needle.size() ) ) {
            ++count;
        }
        return count;
    }
}

int main()
{
    using fheroes2::agent::auditAllMonsters;
    using fheroes2::agent::auditMonster;
    using fheroes2::agent::classifyAbility;
    using fheroes2::agent::classifyWeakness;
    using fheroes2::agent::isSimpleV1Supported;
    using fheroes2::agent::AbilitySemantics;
    using fheroes2::agent::MonsterCapability;

    std::printf( "test_agent_capabilities\n" );

    // The Milestone 1 fixture creatures must be supported.
    check( isSimpleV1Supported( Monster::PEASANT ), "Peasant supported (single-cell melee)" );
    check( isSimpleV1Supported( Monster::ARCHER ), "Archer supported (single-cell shooter)" );
    check( isSimpleV1Supported( Monster::RANGER ), "Ranger supported (double shooting is outcome-only)" );
    check( auditMonster( Monster::RANGER ).hasDoubleShooting, "Ranger audit records double shooting" );
    check( auditMonster( Monster::ARCHER ).isShooter, "Archer audit records shooter" );

    // Action-space-changing creatures must be excluded, each for its audited reason.
    check( !isSimpleV1Supported( Monster::CAVALRY ) && auditMonster( Monster::CAVALRY ).isWide, "Cavalry excluded (wide)" );
    check( !isSimpleV1Supported( Monster::PHOENIX ), "Phoenix excluded (wide/flying)" );
    check( !isSimpleV1Supported( Monster::HYDRA ) && auditMonster( Monster::HYDRA ).hasAllAdjacentMeleeAttack, "Hydra excluded (all-adjacent melee)" );
    check( !isSimpleV1Supported( Monster::LICH ) && auditMonster( Monster::LICH ).hasAreaShot, "Lich excluded (area shot)" );

    // Invalid ids are rejected as invalid rather than crashing.
    check( !auditMonster( 0 ).isValid && !auditMonster( 0 ).simpleV1Supported, "UNKNOWN monster is invalid" );

    {
        const std::vector<MonsterCapability> all = auditAllMonsters();
        check( all.size() == static_cast<size_t>( Monster::MONSTER_COUNT ) - 1, "audit covers every monster id" );

        bool reasonsPresent = true;
        bool supportImpliesValid = true;
        size_t supportedCount = 0;
        for ( const MonsterCapability & r : all ) {
            if ( r.reason.empty() ) {
                reasonsPresent = false;
            }
            if ( r.simpleV1Supported ) {
                ++supportedCount;
                if ( !r.isValid || r.isWide || r.isFlying || r.hasTwoCellMeleeAttack || r.hasAllAdjacentMeleeAttack || r.hasAreaShot ) {
                    supportImpliesValid = false;
                }
            }
        }
        check( reasonsPresent, "every record carries a reason" );
        check( supportImpliesValid, "supported records satisfy the allowlist rule" );
        check( supportedCount >= 10, "a usable number of creatures is supported" );
        std::printf( "        (supported creatures: %zu of %zu)\n", supportedCount, all.size() );
    }

    // Layer-2 semantic adapter: every raw record maps into the closed vocabulary, and the
    // grounded "other" escape hatch is never needed for shipped engine data.
    {
        const std::initializer_list<const char *> triggers = { "always", "on_attack", "on_defense", "on_turn" };
        const std::initializer_list<const char *> targets = { "self", "enemy_unit", "all_adjacent", "all_enemies", "spell_class" };
        const std::initializer_list<const char *> effects
            = { "damage_mult", "resist", "immunity", "spell_cast", "stat_mod", "movement", "attack_shape", "retaliation_mod", "other" };
        const std::initializer_list<const char *> magnitudes = { "percent", "spell_id", "flat", "none" };

        bool vocabularyHolds = true;
        bool noOtherInData = true;
        bool spellIdsResolve = true;
        size_t abilityRecords = 0;
        size_t weaknessRecords = 0;

        for ( int id = Monster::UNKNOWN + 1; id < Monster::MONSTER_COUNT; ++id ) {
            const fheroes2::MonsterData & data = fheroes2::getMonsterData( id );
            for ( const fheroes2::MonsterAbility & ability : data.battleStats.abilities ) {
                ++abilityRecords;
                const AbilitySemantics semantics = classifyAbility( ability.type );
                if ( !oneOf( semantics.trigger, triggers ) || !oneOf( semantics.target, targets ) || !oneOf( semantics.effect, effects )
                     || !oneOf( semantics.magnitudeKind, magnitudes ) ) {
                    vocabularyHolds = false;
                }
                if ( semantics.effect != nullptr && std::strcmp( semantics.effect, "other" ) == 0 ) {
                    noOtherInData = false;
                }
                // A record whose payload is declared to be a spell id must carry a real one.
                if ( semantics.magnitudeKind != nullptr && std::strcmp( semantics.magnitudeKind, "spell_id" ) == 0
                     && !Spell( static_cast<int>( ability.value ) ).isValid() ) {
                    spellIdsResolve = false;
                }
            }
            for ( const fheroes2::MonsterWeakness & weakness : data.battleStats.weaknesses ) {
                ++weaknessRecords;
                const AbilitySemantics semantics = classifyWeakness( weakness.type );
                if ( !oneOf( semantics.trigger, triggers ) || !oneOf( semantics.target, targets ) || !oneOf( semantics.effect, effects )
                     || !oneOf( semantics.magnitudeKind, magnitudes ) ) {
                    vocabularyHolds = false;
                }
                if ( semantics.effect != nullptr && std::strcmp( semantics.effect, "other" ) == 0 ) {
                    noOtherInData = false;
                }
                if ( semantics.magnitudeKind != nullptr && std::strcmp( semantics.magnitudeKind, "spell_id" ) == 0
                     && !Spell( static_cast<int>( weakness.value ) ).isValid() ) {
                    spellIdsResolve = false;
                }
            }
        }

        check( vocabularyHolds, "every ability/weakness maps into the closed vocabulary" );
        check( noOtherInData, "no shipped engine record needs the 'other' escape hatch" );
        check( spellIdsResolve, "every spell_id payload is a valid engine spell" );
        std::printf( "        (classified records: %zu abilities, %zu weaknesses)\n", abilityRecords, weaknessRecords );
    }

    // Known monsters whose engine semantics were verified at the cited call sites.
    {
        // Cyclops: 20% chance to Paralyze on attack (Battle::Unit::GetSpellMagic).
        const fheroes2::MonsterAbility * caster = findAbility( Monster::CYCLOPS, fheroes2::MonsterAbilityType::SPELL_CASTER );
        check( caster != nullptr && semanticsEqual( classifyAbility( caster->type ), "on_attack", "enemy_unit", "spell_cast", "spell_id" )
                   && static_cast<int>( caster->value ) == Spell::PARALYZE && caster->percentage == 20,
               "Cyclops spell caster maps to on_attack spell_cast carrying Spell::PARALYZE" );

        // Crusader: double damage against undead (Battle::Unit::CalculateDamageUnit).
        const fheroes2::MonsterAbility * slayer = findAbility( Monster::CRUSADER, fheroes2::MonsterAbilityType::DOUBLE_DAMAGE_TO_UNDEAD );
        check( slayer != nullptr && semanticsEqual( classifyAbility( slayer->type ), "on_attack", "enemy_unit", "damage_mult", "none" ),
               "Crusader double damage to undead maps to on_attack damage_mult" );

        // Giant: immune to mind spells (getSpellResistance).
        const fheroes2::MonsterAbility * mind = findAbility( Monster::GIANT, fheroes2::MonsterAbilityType::MIND_SPELL_IMMUNITY );
        check( mind != nullptr && semanticsEqual( classifyAbility( mind->type ), "always", "spell_class", "immunity", "none" ),
               "Giant mind-spell immunity maps to always spell_class immunity" );

        // Bone Dragon: -1 morale for every enemy unit (Battle::Unit::GetMorale).
        const fheroes2::MonsterAbility * dread = findAbility( Monster::BONE_DRAGON, fheroes2::MonsterAbilityType::MORAL_DECREMENT );
        check( dread != nullptr && semanticsEqual( classifyAbility( dread->type ), "always", "all_enemies", "stat_mod", "flat" ) && dread->percentage == 100
                   && dread->value == 1,
               "Bone Dragon morale decrement maps to always all_enemies stat_mod, flat 1" );

        // Genie: 10% chance to halve the enemy stack (Battle::Arena::GetTargetsForDamage).
        const fheroes2::MonsterAbility * halving = findAbility( Monster::GENIE, fheroes2::MonsterAbilityType::ENEMY_HALVING );
        check( halving != nullptr && semanticsEqual( classifyAbility( halving->type ), "on_attack", "enemy_unit", "damage_mult", "percent" )
                   && halving->percentage == 10,
               "Genie enemy halving maps to on_attack damage_mult with percent chance" );
    }

    // The exported audit carries the four layer-2 fields on every ability and weakness record.
    {
        char auditPath[] = "/tmp/agent_caps_audit_XXXXXX";
        const int fd = ::mkstemp( auditPath );
        bool fieldsComplete = false;
        if ( fd != -1 ) {
            ::close( fd );
            if ( fheroes2::agent::writeCapabilityAudit( auditPath ) ) {
                std::ifstream in( auditPath );
                std::stringstream buffer;
                buffer << in.rdbuf();
                const std::string audit = buffer.str();

                size_t records = 0;
                for ( int id = Monster::UNKNOWN + 1; id < Monster::MONSTER_COUNT; ++id ) {
                    const fheroes2::MonsterData & data = fheroes2::getMonsterData( id );
                    records += data.battleStats.abilities.size() + data.battleStats.weaknesses.size();
                }

                fieldsComplete = records > 0 && countOccurrences( audit, "\"type_id\": " ) == records
                                 && countOccurrences( audit, "\"trigger\": \"" ) == records && countOccurrences( audit, "\"target\": \"" ) == records
                                 && countOccurrences( audit, "\"effect\": \"" ) == records && countOccurrences( audit, "\"magnitude_kind\": \"" ) == records;
            }
            std::remove( auditPath );
        }
        check( fieldsComplete, "written audit carries trigger/target/effect/magnitude_kind on every record" );
    }

    std::printf( "%d passed, %d failed\n", passed, failed );
    return ( failed == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
