/***************************************************************************
 *   fheroes2 agent tests: observation serialization (ADR 0001 full_v1)     *
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <string>

#include "agent_observation.h"

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

    bool contains( const std::string & haystack, const char * needle )
    {
        return haystack.find( needle ) != std::string::npos;
    }

    fheroes2::agent::ObservedUnit peasantStack( const uint32_t uid, const bool isAttacker, const uint32_t count )
    {
        fheroes2::agent::ObservedUnit unit;
        unit.uid = uid;
        unit.monsterId = 1;
        unit.isAttacker = isAttacker;
        unit.count = count;
        unit.initialCount = count;
        unit.hitPoints = count;
        unit.topHitPoints = 1;
        unit.attack = 1;
        unit.defense = 1;
        unit.speed = 3;
        unit.headCell = 34;
        return unit;
    }
}

int main()
{
    using fheroes2::agent::Observation;
    using fheroes2::agent::ObservedUnit;
    using fheroes2::agent::observationToJson;

    // An observation with no living stacks still has to be well formed, since it is what a
    // terminal or near-terminal board looks like.
    {
        const Observation empty;
        const std::string json = observationToJson( empty );
        check( contains( json, "\"schema\":\"observation_full_v1\"" ), "schema tag is present" );
        check( contains( json, "\"units\":[]" ), "no living stacks serializes as an empty array" );
        check( json.front() == '{' && json.back() == '}', "output is a single JSON object" );
        check( json.find( '\n' ) == std::string::npos, "output is one line, so it is JSONL-safe" );
    }

    // Field presence and typing. A boolean written as 1 rather than true would parse but would
    // silently change the dtype a loader infers, so the spelling is asserted.
    {
        Observation observation;
        observation.engineDecisionIndex = 7;
        observation.round = 2;
        observation.activeUid = 5;
        observation.activeIsAttacker = true;

        ObservedUnit active = peasantStack( 5, true, 40 );
        active.isActive = true;
        active.shots = 12;
        active.isArcher = true;
        observation.units.push_back( active );

        const std::string json = observationToJson( observation );
        check( contains( json, "\"engine_decision_index\":7" ), "decision index is serialized" );
        check( contains( json, "\"round\":2" ), "round is serialized" );
        check( contains( json, "\"active_uid\":5" ), "active uid is serialized" );
        check( contains( json, "\"active_is_attacker\":true" ), "booleans are true, not 1" );
        check( contains( json, "\"side\":\"attacker\"" ), "side is a readable string" );
        check( contains( json, "\"shots\":12" ), "shots are serialized, so a shooter is distinguishable" );
        check( contains( json, "\"archer\":true" ), "archer flag is serialized" );
        check( contains( json, "\"initial_count\":40" ), "initial count is serialized, so losses are derivable" );
    }

    // Negative values occur in normal play: morale and luck run negative, and a single-cell
    // stack has no tail. Unsigned formatting here would produce huge positive numbers.
    {
        Observation observation;
        ObservedUnit unit = peasantStack( 1, false, 10 );
        unit.morale = -1;
        unit.luck = -2;
        unit.tailCell = -1;
        observation.units.push_back( unit );

        const std::string json = observationToJson( observation );
        check( contains( json, "\"morale\":-1" ), "negative morale is signed" );
        check( contains( json, "\"luck\":-2" ), "negative luck is signed" );
        check( contains( json, "\"tail_cell\":-1" ), "absent tail cell is -1, not a huge unsigned" );
    }

    // Determinism, which is what lets two runs of one scenario be compared byte for byte.
    {
        Observation a;
        a.activeUid = 3;
        a.units.push_back( peasantStack( 3, true, 20 ) );
        a.units.push_back( peasantStack( 4, false, 20 ) );

        Observation b = a;
        check( observationToJson( a ) == observationToJson( b ), "equal observations serialize identically" );

        Observation c = a;
        c.units[1].count = 19;
        check( observationToJson( a ) != observationToJson( c ), "one changed creature count changes the bytes" );

        Observation d = a;
        d.units[1].headCell = 35;
        check( observationToJson( a ) != observationToJson( d ), "one moved stack changes the bytes" );
    }

    // Field aliasing. Two different states must not collide, which is the same property the
    // digest writer asserts with length prefixes.
    {
        Observation a;
        a.units.push_back( peasantStack( 1, true, 12 ) );
        a.units[0].hitPoints = 3;

        Observation b;
        b.units.push_back( peasantStack( 1, true, 3 ) );
        b.units[0].hitPoints = 12;

        check( observationToJson( a ) != observationToJson( b ), "swapped count and hit points do not alias" );
    }

    std::printf( "%d passed, %d failed\n", passed, failed );
    return ( failed == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
