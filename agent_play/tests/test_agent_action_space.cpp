/***************************************************************************
 *   fheroes2 agent tests: canonical simple_v1 action indexing (ADR 0002)   *
 *                                                                          *
 *   Pure indexing math only; enumeration against a live arena is covered   *
 *   by verify_m3.sh through the worker's --audit-coverage mode.            *
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "agent_action_space.h"

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
    using namespace fheroes2::agent;

    std::printf( "test_agent_action_space\n" );

    check( actionSpaceSize == 793, "action space size is 793" );
    check( actionSkipIndex == 0 && actionMoveBase == 1 && actionRangedBase == 100 && actionMeleeBase == 199, "section bases" );
    check( actionMeleeBase + 99 * 6 == actionSpaceSize, "melee section ends exactly at the space size" );

    {
        bool ok = true;
        for ( int i = 0; i < 6; ++i ) {
            const Battle::CellDirection dir = meleeDirectionFromIndex( i );
            if ( meleeDirectionIndex( dir ) != i ) {
                ok = false;
            }
        }
        check( ok, "direction index round-trips for all six directions" );
        check( meleeDirectionIndex( Battle::CellDirection::UNKNOWN ) == -1, "UNKNOWN direction has no melee index" );
        check( meleeDirectionIndex( Battle::CellDirection::CENTER ) == -1, "CENTER direction has no melee index" );
        check( meleeDirectionFromIndex( 6 ) == Battle::CellDirection::UNKNOWN, "out-of-range index decodes to UNKNOWN" );
    }

    {
        // Every (targetCell, direction) pair maps to a unique in-range melee index.
        bool unique = true;
        bool inRange = true;
        std::vector<uint8_t> seen( actionSpaceSize, 0 );
        for ( uint32_t t = 0; t < 99; ++t ) {
            for ( uint32_t d = 0; d < 6; ++d ) {
                const uint32_t index = actionMeleeBase + t * 6 + d;
                if ( index < actionMeleeBase || index >= actionSpaceSize ) {
                    inRange = false;
                }
                else {
                    if ( seen[index] != 0 ) {
                        unique = false;
                    }
                    seen[index] = 1;
                }
            }
        }
        check( inRange, "all melee indices are in range" );
        check( unique, "melee indices are unique" );

        bool movesDisjoint = true;
        for ( uint32_t c = 0; c < 99; ++c ) {
            if ( seen[actionMoveBase + c] != 0 || seen[actionRangedBase + c] != 0 ) {
                movesDisjoint = false;
            }
        }
        check( movesDisjoint && seen[actionSkipIndex] == 0, "sections do not overlap" );
    }

    check( std::strcmp( candidateTypeName( CandidateType::Skip ), "skip" ) == 0 && std::strcmp( candidateTypeName( CandidateType::Move ), "move" ) == 0
               && std::strcmp( candidateTypeName( CandidateType::RangedAttack ), "ranged" ) == 0
               && std::strcmp( candidateTypeName( CandidateType::MeleeAttack ), "melee" ) == 0,
           "candidate type names" );

    std::printf( "%d passed, %d failed\n", passed, failed );
    return ( failed == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
