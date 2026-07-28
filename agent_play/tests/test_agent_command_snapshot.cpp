/***************************************************************************
 *   fheroes2 agent tests: CommandSnapshot decoding                         *
 *                                                                          *
 *   Battle::Command stores parameters reversed and GetNextValue() pops     *
 *   from the back (spec §3.8), so the decoder must yield semantic order    *
 *   and must never consume the original command (spec §10.5).              *
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "agent_command_snapshot.h"
#include "agent_trajectory.h"
#include "battle_command.h"

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
    using fheroes2::agent::canonicalCommandKey;
    using fheroes2::agent::CommandSnapshot;
    using fheroes2::agent::snapshotCommand;

    std::printf( "test_agent_command_snapshot\n" );

    {
        const Battle::Command command( Battle::Command::MOVE, 7, 34 );
        // Raw storage is reversed; the decoder must undo that.
        check( command.size() == 2 && command[0] == 34 && command[1] == 7, "MOVE raw storage is reversed" );

        const CommandSnapshot snapshot = snapshotCommand( command );
        check( snapshot.type == Battle::CommandType::MOVE && snapshot.unitUid == 7 && snapshot.moveCell == 34, "MOVE decodes semantic fields" );
        check( snapshot.params == std::vector<int>( { 7, 34 } ), "MOVE params are in semantic order" );
        check( canonicalCommandKey( snapshot ) == "move:7:34", "MOVE canonical key" );

        // The original command must not be consumed by taking a snapshot.
        check( command.size() == 2 && command.GetType() == Battle::CommandType::MOVE, "original command is not consumed" );

        const CommandSnapshot again = snapshotCommand( command );
        check( again.params == snapshot.params, "snapshotting twice yields identical results" );
    }

    {
        const Battle::Command command( Battle::Command::ATTACK, 1, 6, 34, 45, 3 );
        const CommandSnapshot snapshot = snapshotCommand( command );
        check( snapshot.type == Battle::CommandType::ATTACK && snapshot.unitUid == 1 && snapshot.defenderUid == 6 && snapshot.moveCell == 34
                   && snapshot.targetCell == 45 && snapshot.direction == 3,
               "ATTACK decodes all five semantic fields" );
        check( canonicalCommandKey( snapshot ) == "attack:1:6:34:45:3", "ATTACK canonical key" );
    }

    {
        const Battle::Command command( Battle::Command::SKIP, 9 );
        const CommandSnapshot snapshot = snapshotCommand( command );
        check( snapshot.type == Battle::CommandType::SKIP && snapshot.unitUid == 9, "SKIP decodes the unit UID" );
        check( canonicalCommandKey( snapshot ) == "skip:9", "SKIP canonical key" );
    }

    {
        const Battle::Command command( Battle::Command::MORALE, 5, true );
        const CommandSnapshot snapshot = snapshotCommand( command );
        check( snapshot.type == Battle::CommandType::MORALE && snapshot.unitUid == 5 && snapshot.moraleIsGood, "MORALE decodes UID and polarity" );
        check( canonicalCommandKey( snapshot ) == "morale:5:1", "MORALE canonical key" );
    }

    {
        using fheroes2::agent::computeDecisionDigest;
        using fheroes2::agent::DecisionRecord;

        DecisionRecord decision;
        decision.engineDecisionIndex = 1;
        decision.unitUid = 7;
        decision.actions.push_back( snapshotCommand( Battle::Command( Battle::Command::MOVE, 7, 34 ) ) );

        const std::vector<DecisionRecord> streamA( 1, decision );
        std::vector<DecisionRecord> streamB( 1, decision );

        check( computeDecisionDigest( streamA ) == computeDecisionDigest( streamB ), "identical decision streams digest identically" );
        check( computeDecisionDigest( streamA ).size() == 64, "decision digest is a SHA-256 hex string" );

        streamB[0].actions[0].params[1] = 35; // move to a different cell
        check( computeDecisionDigest( streamA ) != computeDecisionDigest( streamB ), "changing one parameter changes the digest" );

        check( computeDecisionDigest( {} ) != computeDecisionDigest( streamA ), "empty stream digests differently" );
    }

    std::printf( "%d passed, %d failed\n", passed, failed );
    return ( failed == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
