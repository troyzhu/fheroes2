/***************************************************************************
 *   fheroes2 agent tests: monster capability audit and simple_v1 allowlist *
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>

#include "agent_capabilities.h"
#include "monster.h"

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
}

int main()
{
    using fheroes2::agent::auditAllMonsters;
    using fheroes2::agent::auditMonster;
    using fheroes2::agent::isSimpleV1Supported;
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

    std::printf( "%d passed, %d failed\n", passed, failed );
    return ( failed == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
