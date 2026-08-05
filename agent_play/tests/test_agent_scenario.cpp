/***************************************************************************
 *   fheroes2 agent tests: scenario validation and Milestone 1 fixtures     *
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <string>

#include "agent_scenario.h"
#include "ground.h"
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

    fheroes2::agent::Scenario validBase()
    {
        fheroes2::agent::Scenario s;
        s.scenarioId = "test_base";
        s.groundType = Maps::Ground::GRASS;
        s.worldSeed = 1;
        s.attacker.slots[0] = { Monster::PEASANT, 10 };
        s.defender.slots[0] = { Monster::PEASANT, 10 };
        return s;
    }
}

int main()
{
    using fheroes2::agent::milestone1Fixtures;
    using fheroes2::agent::Scenario;
    using fheroes2::agent::validateScenario;

    std::printf( "test_agent_scenario\n" );

    check( validateScenario( validBase() ).empty(), "base scenario is valid" );

    {
        const auto & fixtures = milestone1Fixtures();
        check( fixtures.size() == 5, "five Milestone 1 fixtures" );
        bool allValid = true;
        for ( const Scenario & s : fixtures ) {
            const std::string error = validateScenario( s );
            if ( !error.empty() ) {
                std::printf( "        fixture rejected: %s\n", error.c_str() );
                allValid = false;
            }
        }
        check( allValid, "every Milestone 1 fixture passes validation" );
    }

    {
        Scenario s = validBase();
        s.scenarioId.clear();
        check( !validateScenario( s ).empty(), "empty scenario id is rejected" );
    }
    {
        Scenario s = validBase();
        s.tileIndex = 2;
        check( !validateScenario( s ).empty(), "tile index other than 1 is rejected" );
    }
    {
        Scenario s = validBase();
        s.groundType = Maps::Ground::UNKNOWN;
        check( !validateScenario( s ).empty(), "UNKNOWN ground is rejected" );
    }
    {
        Scenario s = validBase();
        s.groundType = Maps::Ground::GRASS | Maps::Ground::DIRT;
        check( !validateScenario( s ).empty(), "multi-bit ground mask is rejected" );
    }
    {
        Scenario s = validBase();
        s.maxRounds = 0;
        check( !validateScenario( s ).empty(), "zero max rounds is rejected" );
    }
    {
        Scenario s = validBase();
        s.attacker.slots[0] = {};
        check( !validateScenario( s ).empty(), "side without stacks is rejected" );
    }
    {
        Scenario s = validBase();
        s.attacker.slots[1] = { 0, 5 }; // UNKNOWN monster with a nonzero count
        check( !validateScenario( s ).empty(), "invalid monster id is rejected" );
    }
    {
        Scenario s = validBase();
        s.defender.slots[4] = { Monster::PEASANT, fheroes2::agent::scenarioMaxStackCount + 1 };
        check( !validateScenario( s ).empty(), "count above the safety maximum is rejected" );
    }
    {
        Scenario s = validBase();
        s.attackerCommander = { true, 13, 12 };
        check( validateScenario( s ).empty(), "a commander with sane stats is accepted" );
    }
    {
        Scenario s = validBase();
        s.defenderCommander = { true, -1, 5 };
        check( !validateScenario( s ).empty(), "a commander with negative attack is rejected" );
    }
    {
        Scenario s = validBase();
        s.attackerCommander = { true, 0, fheroes2::agent::scenarioMaxCommanderStat + 1 };
        check( !validateScenario( s ).empty(), "a commander stat above the cap is rejected" );
    }
    {
        // Stats without the flag mean a caller forgot to mark the commander present, and the
        // battle would silently run without the stats they specified.
        Scenario s = validBase();
        s.attackerCommander = { false, 13, 12 };
        check( !validateScenario( s ).empty(), "commander stats without the present flag are rejected" );
    }

    std::printf( "%d passed, %d failed\n", passed, failed );
    return ( failed == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
