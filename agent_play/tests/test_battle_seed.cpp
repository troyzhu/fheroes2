/***************************************************************************
 *   fheroes2 agent tests: Battle::computeBattleSeed                        *
 *                                                                          *
 *   Spec §7.3: the shared combat-seed helper is part of the trajectory     *
 *   compatibility contract, so it gets golden-value tests with fixed slot  *
 *   layouts, including empty slots.                                        *
 *                                                                          *
 *   The four golden values below were produced by the engine itself (the   *
 *   Phase 0 spike linked against unmodified game objects, world seed       *
 *   20260726 -> map seed 2227197244) and recorded in                       *
 *   docs/agent/benchmark_m2.md. If one of these asserts ever fires, the    *
 *   helper no longer reproduces historical battles: treat it as a          *
 *   compatibility break, not a test to update casually.                    *
 ***************************************************************************/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "army.h"
#include "army_troop.h"
#include "battle_seed.h"
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

    struct SlotSpec
    {
        size_t slot{ 0 };
        int monsterId{ 0 };
        uint32_t count{ 0 };
    };

    void fillArmy( Army & army, const std::vector<SlotSpec> & slots )
    {
        army.Reset( false );
        for ( const SlotSpec & s : slots ) {
            army.GetTroop( s.slot )->Set( Monster( s.monsterId ), s.count );
        }
    }

    uint32_t seedFor( const int32_t tileIndex, const uint32_t mapSeed, const std::vector<SlotSpec> & attackerSlots, const std::vector<SlotSpec> & defenderSlots )
    {
        Army attacker;
        Army defender;
        fillArmy( attacker, attackerSlots );
        fillArmy( defender, defenderSlots );
        return Battle::computeBattleSeed( tileIndex, mapSeed, attacker, defender );
    }
}

int main()
{
    const uint32_t mapSeed = 2227197244U; // derived from world seed 20260726, see benchmark_m2.md

    std::printf( "test_battle_seed\n" );

    // Golden values recorded from engine behaviour before the helper was extracted.
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, { { 0, Monster::PEASANT, 50 } } ) == 1356111745U, "golden: Peasant 50 vs Peasant 50" );
    check( seedFor( 1, mapSeed, { { 0, Monster::ARCHER, 20 } }, { { 0, Monster::PEASANT, 60 } } ) == 1381489788U, "golden: Archer 20 vs Peasant 60" );
    check( seedFor( 1, mapSeed, { { 0, Monster::RANGER, 100 } }, { { 0, Monster::RANGER, 100 } } ) == 1274517553U, "golden: Ranger 100 vs Ranger 100" );
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 1000 } }, { { 0, Monster::PEASANT, 1000 } } ) == 3437871903U, "golden: Peasant 1000 vs Peasant 1000" );

    // Determinism of the pure function itself.
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, {} ) == seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, {} ),
           "identical inputs produce identical seeds" );

    // Empty slots are folded positionally: the same troop in a different slot must change the seed.
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, {} ) != seedFor( 1, mapSeed, { { 1, Monster::PEASANT, 50 } }, {} ),
           "slot position matters (slot 0 vs slot 1)" );
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, {} ) != seedFor( 1, mapSeed, { { 4, Monster::PEASANT, 50 } }, {} ),
           "slot position matters (slot 0 vs slot 4)" );

    // Attacker and defender armies are folded in order, so swapping sides changes the seed.
    check( seedFor( 1, mapSeed, { { 0, Monster::ARCHER, 20 } }, { { 0, Monster::PEASANT, 60 } } )
               != seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 60 } }, { { 0, Monster::ARCHER, 20 } } ),
           "attacker/defender order matters" );

    // Every input component participates.
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, { { 0, Monster::PEASANT, 50 } } )
               != seedFor( 2, mapSeed, { { 0, Monster::PEASANT, 50 } }, { { 0, Monster::PEASANT, 50 } } ),
           "tile index matters" );
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, { { 0, Monster::PEASANT, 50 } } )
               != seedFor( 1, mapSeed + 1, { { 0, Monster::PEASANT, 50 } }, { { 0, Monster::PEASANT, 50 } } ),
           "map seed matters" );
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, { { 0, Monster::PEASANT, 50 } } )
               != seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 51 } }, { { 0, Monster::PEASANT, 50 } } ),
           "troop count matters" );
    check( seedFor( 1, mapSeed, { { 0, Monster::PEASANT, 50 } }, { { 0, Monster::PEASANT, 50 } } )
               != seedFor( 1, mapSeed, { { 0, Monster::ARCHER, 50 } }, { { 0, Monster::PEASANT, 50 } } ),
           "monster id matters" );

    // Five-slot layouts with interior empty slots stay deterministic and distinct.
    const std::vector<SlotSpec> spread = { { 0, Monster::PEASANT, 10 }, { 2, Monster::ARCHER, 5 }, { 4, Monster::RANGER, 3 } };
    const std::vector<SlotSpec> packed = { { 0, Monster::PEASANT, 10 }, { 1, Monster::ARCHER, 5 }, { 2, Monster::RANGER, 3 } };
    check( seedFor( 1, mapSeed, spread, {} ) == seedFor( 1, mapSeed, spread, {} ), "sparse five-slot layout is deterministic" );
    check( seedFor( 1, mapSeed, spread, {} ) != seedFor( 1, mapSeed, packed, {} ), "sparse vs packed layouts differ" );

    std::printf( "%d passed, %d failed\n", passed, failed );
    return ( failed == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
