/***************************************************************************
 *   fheroes2 agent Phase 0 spike: headless creature-only battle smoke test *
 *                                                                          *
 *   Answers the Phase 0 questions in the agent system spec, Section 2.4:    *
 *     - can a Battle::Arena be constructed and run with no UI and no        *
 *       display/audio/AGG initialization?                                   *
 *     - is the outcome deterministic across repeated identical runs?        *
 *     - can one process run many fresh arenas sequentially?                 *
 *     - what are the effective world and combat seeds?                      *
 *                                                                           *
 *   This is a throwaway diagnostic, not the eventual agent worker. It is     *
 *   deliberately self-contained so it can be built by relinking the already- *
 *   compiled game objects (see build_spike.sh).                              *
 ***************************************************************************/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "army.h"
#include "army_troop.h"
#include "battle.h"
#include "battle_arena.h"
#include "battle_army.h"
#include "battle_seed.h"
#include "battle_troop.h"
#include "color.h"
#include "ground.h"
#include "logging.h"
#include "monster.h"
#include "players.h"
#include "race.h"
#include "rand.h"
#include "settings.h"
#include "world.h"

namespace
{
    // NOTE: this spike originally carried a verbatim copy of the anonymous-namespace
    // computeBattleSeed() from src/fheroes2/battle/battle_main.cpp, because the engine did not
    // export it. That duplicate proved the engine's seed was reproducible externally and became
    // the argument for the shared helper the spec asked for; the spike now calls the extracted
    // Battle::computeBattleSeed() from battle_seed.h like the engine itself does.

    struct EpisodeResult
    {
        uint32_t worldSeed{ 0 };
        uint32_t mapSeed{ 0 };
        uint32_t combatSeed{ 0 };
        int32_t rounds{ 0 };
        uint32_t attackerResult{ 0 };
        uint32_t defenderResult{ 0 };
        uint32_t attackerRemaining{ 0 };
        uint32_t defenderRemaining{ 0 };
        // Terminal battle-side state, read from the Force objects BEFORE arena destruction.
        // The input Army objects are deliberately not synchronized by the engine outside
        // Battle::Loader, so they cannot be used as the terminal observation.
        uint32_t attackerLiveStacks{ 0 };
        uint32_t defenderLiveStacks{ 0 };
        uint32_t attackerLiveCreatures{ 0 };
        uint32_t defenderLiveCreatures{ 0 };
        uint32_t attackerHitPoints{ 0 };
        uint32_t defenderHitPoints{ 0 };
        uint64_t forceFold{ 1469598103934665603ULL };
        std::string digest;
    };

    // Walks a battle Force and accumulates both summary counters and an order-sensitive fold over
    // per-unit state. This is the minimal version of the observation serializer in spec Section 12.
    void summarizeForce( const Battle::Force & force, uint32_t & liveStacks, uint32_t & liveCreatures, uint32_t & hitPoints, uint64_t & fold )
    {
        const auto mix = [&fold]( const uint64_t v ) {
            for ( int i = 0; i < 8; ++i ) {
                fold ^= ( v >> ( i * 8 ) ) & 0xFF;
                fold *= 1099511628211ULL;
            }
        };

        for ( const Battle::Unit * unit : force ) {
            if ( unit == nullptr ) {
                continue;
            }

            mix( unit->GetUID() );
            mix( static_cast<uint64_t>( unit->GetID() ) );
            mix( unit->GetCount() );
            mix( unit->GetHitPoints() );
            mix( static_cast<uint64_t>( unit->GetHeadIndex() ) );
            mix( unit->isValid() ? 1U : 0U );

            if ( unit->isValid() ) {
                ++liveStacks;
                liveCreatures += unit->GetCount();
                hitPoints += unit->GetHitPoints();
            }
        }
    }

    // A cheap order-sensitive digest. Not SHA-256: this spike only needs to detect divergence
    // between runs, and avoiding a hash dependency keeps the relink trivial.
    std::string makeDigest( const EpisodeResult & r )
    {
        uint64_t h = 1469598103934665603ULL;
        const auto fold = [&h]( const uint64_t v ) {
            for ( int i = 0; i < 8; ++i ) {
                h ^= ( v >> ( i * 8 ) ) & 0xFF;
                h *= 1099511628211ULL;
            }
        };

        fold( r.mapSeed );
        fold( r.combatSeed );
        fold( static_cast<uint64_t>( r.rounds ) );
        fold( r.attackerResult );
        fold( r.defenderResult );
        fold( r.attackerLiveStacks );
        fold( r.defenderLiveStacks );
        fold( r.attackerLiveCreatures );
        fold( r.defenderLiveCreatures );
        fold( r.attackerHitPoints );
        fold( r.defenderHitPoints );
        fold( r.forceFold );

        char buf[32];
        std::snprintf( buf, sizeof( buf ), "%016llx", static_cast<unsigned long long>( h ) );
        return std::string( buf );
    }

    uint32_t totalCount( const Army & army )
    {
        uint32_t total = 0;
        for ( size_t i = 0; i < army.Size(); ++i ) {
            const Troop * troop = army.GetTroop( i );
            if ( troop != nullptr && troop->isValid() ) {
                total += troop->GetCount();
            }
        }
        return total;
    }

    EpisodeResult runEpisode( const uint32_t worldSeed, const int monsterA, const uint32_t countA, const int monsterB, const uint32_t countB,
                              const int32_t maxRounds, const bool seedGlobalRng )
    {
        EpisodeResult result;
        result.worldSeed = worldSeed;

        // Phase 0 experiment: World::Defaults() sets its map seed from the global Rand::Get(),
        // which draws from a thread-local PCG32 exposed by reference. If reseeding that device
        // makes the map seed reproducible, no engine change is needed for deterministic worlds.
        if ( seedGlobalRng ) {
            Rand::CurrentThreadRandomDevice() = Rand::PCG32( worldSeed );
        }

        world.generateBattleOnlyMap( Maps::Ground::GRASS );
        result.mapSeed = world.GetMapSeed();

        Settings & conf = Settings::Get();
        conf.GetPlayers().Init( static_cast<PlayerColorsSet>( PlayerColor::BLUE ) | static_cast<PlayerColorsSet>( PlayerColor::RED ) );
        world.InitKingdoms();

        Players::SetPlayerRace( PlayerColor::BLUE, Race::KNGT );
        Players::SetPlayerControl( PlayerColor::BLUE, CONTROL_AI );
        Players::SetPlayerRace( PlayerColor::RED, Race::KNGT );
        Players::SetPlayerControl( PlayerColor::RED, CONTROL_AI );

        Army attackingArmy;
        attackingArmy.Reset( false );
        attackingArmy.SetColor( PlayerColor::BLUE );
        attackingArmy.GetTroop( 0 )->Set( Monster( monsterA ), countA );

        Army defendingArmy;
        defendingArmy.Reset( false );
        defendingArmy.SetColor( PlayerColor::RED );
        defendingArmy.GetTroop( 0 )->Set( Monster( monsterB ), countB );

        const int32_t tileIndex = 1;
        result.combatSeed = Battle::computeBattleSeed( tileIndex, result.mapSeed, attackingArmy, defendingArmy );

        Rand::PCG32 randomGenerator( result.combatSeed );

        {
            Battle::Arena arena( attackingArmy, defendingArmy, tileIndex, false, randomGenerator );

            while ( arena.BattleValid() && result.rounds < maxRounds ) {
                arena.Turns();
                ++result.rounds;
            }

            const Battle::Result & battleResult = arena.GetResult();
            result.attackerResult = battleResult.attacker;
            result.defenderResult = battleResult.defender;

            // Must happen before the arena leaves scope.
            summarizeForce( arena.getAttackingForce(), result.attackerLiveStacks, result.attackerLiveCreatures, result.attackerHitPoints, result.forceFold );
            summarizeForce( arena.getDefendingForce(), result.defenderLiveStacks, result.defenderLiveCreatures, result.defenderHitPoints, result.forceFold );
        }
        // Arena destroyed here: the engine keeps a file-static single-arena pointer, so this
        // scope exit is what makes the next episode legal.

        result.attackerRemaining = totalCount( attackingArmy );
        result.defenderRemaining = totalCount( defendingArmy );
        result.digest = makeDigest( result );

        return result;
    }
}

int main( int argc, char ** argv )
{
    int episodes = 1;
    uint32_t worldSeed = 20260726;
    int monsterA = Monster::PEASANT;
    int monsterB = Monster::PEASANT;
    uint32_t countA = 50;
    uint32_t countB = 50;
    int32_t maxRounds = 200;
    bool seedGlobalRng = true;
    bool quiet = false;

    for ( int i = 1; i < argc; ++i ) {
        const auto next = [&]( const char * name ) -> const char * {
            if ( i + 1 >= argc ) {
                std::fprintf( stderr, "missing value for %s\n", name );
                std::exit( 2 );
            }
            return argv[++i];
        };

        if ( std::strcmp( argv[i], "--episodes" ) == 0 ) {
            episodes = std::atoi( next( "--episodes" ) );
        }
        else if ( std::strcmp( argv[i], "--world-seed" ) == 0 ) {
            worldSeed = static_cast<uint32_t>( std::strtoul( next( "--world-seed" ), nullptr, 10 ) );
        }
        else if ( std::strcmp( argv[i], "--monster-a" ) == 0 ) {
            monsterA = std::atoi( next( "--monster-a" ) );
        }
        else if ( std::strcmp( argv[i], "--monster-b" ) == 0 ) {
            monsterB = std::atoi( next( "--monster-b" ) );
        }
        else if ( std::strcmp( argv[i], "--count-a" ) == 0 ) {
            countA = static_cast<uint32_t>( std::atoi( next( "--count-a" ) ) );
        }
        else if ( std::strcmp( argv[i], "--count-b" ) == 0 ) {
            countB = static_cast<uint32_t>( std::atoi( next( "--count-b" ) ) );
        }
        else if ( std::strcmp( argv[i], "--max-rounds" ) == 0 ) {
            maxRounds = std::atoi( next( "--max-rounds" ) );
        }
        else if ( std::strcmp( argv[i], "--no-global-seed" ) == 0 ) {
            seedGlobalRng = false;
        }
        else if ( std::strcmp( argv[i], "--quiet" ) == 0 ) {
            quiet = true;
        }
        else {
            std::fprintf( stderr, "unknown argument: %s\n", argv[i] );
            return 2;
        }
    }

    Logging::InitLog();

    std::fprintf( stderr, "[spike] episodes=%d world_seed=%u monsters=%d/%d counts=%u/%u global_seed=%s\n", episodes, worldSeed, monsterA, monsterB, countA,
                  countB, seedGlobalRng ? "on" : "off" );

    std::vector<std::string> digests;
    digests.reserve( static_cast<size_t>( episodes ) );

    for ( int e = 0; e < episodes; ++e ) {
        const EpisodeResult r = runEpisode( worldSeed, monsterA, countA, monsterB, countB, maxRounds, seedGlobalRng );
        digests.push_back( r.digest );

        if ( !quiet ) {
            std::printf( "episode=%d map_seed=%u combat_seed=%u rounds=%d winner=%s a_stacks=%u a_creatures=%u a_hp=%u d_stacks=%u d_creatures=%u d_hp=%u "
                         "army_synced=%s digest=%s\n",
                         e, r.mapSeed, r.combatSeed, r.rounds,
                         ( ( r.attackerResult & Battle::RESULT_WINS ) ? "attacker" : ( ( r.defenderResult & Battle::RESULT_WINS ) ? "defender" : "none" ) ),
                         r.attackerLiveStacks, r.attackerLiveCreatures, r.attackerHitPoints, r.defenderLiveStacks, r.defenderLiveCreatures, r.defenderHitPoints,
                         ( ( r.attackerRemaining != countA || r.defenderRemaining != countB ) ? "yes" : "no" ), r.digest.c_str() );
        }
    }

    size_t distinct = 0;
    for ( size_t i = 0; i < digests.size(); ++i ) {
        bool seen = false;
        for ( size_t j = 0; j < i; ++j ) {
            if ( digests[i] == digests[j] ) {
                seen = true;
                break;
            }
        }
        if ( !seen ) {
            ++distinct;
        }
    }

    std::printf( "SUMMARY episodes=%d distinct_digests=%zu deterministic=%s\n", episodes, distinct, ( distinct == 1 ? "yes" : "no" ) );

    return 0;
}
