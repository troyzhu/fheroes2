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

#include "agent_trajectory.h"

#include "agent_battle_runner.h"
#include "agent_digest.h"
#include "agent_scenario.h"

namespace
{
    // Minimal JSON string escaping. Only used for strings this project itself produces
    // (scenario ids, digests, type names), but kept correct for arbitrary input anyway.
    std::string escapeJson( const std::string & text )
    {
        std::string result;
        result.reserve( text.size() );
        for ( const char ch : text ) {
            switch ( ch ) {
            case '"':
                result += "\\\"";
                break;
            case '\\':
                result += "\\\\";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                if ( static_cast<unsigned char>( ch ) < 0x20 ) {
                    static const char * hexDigits = "0123456789abcdef";
                    result += "\\u00";
                    result += hexDigits[( static_cast<unsigned char>( ch ) >> 4 ) & 0xF];
                    result += hexDigits[static_cast<unsigned char>( ch ) & 0xF];
                }
                else {
                    result += ch;
                }
                break;
            }
        }
        return result;
    }

    std::string sideToJson( const fheroes2::agent::SideSpec & side )
    {
        std::string json = "[";
        bool first = true;
        for ( const fheroes2::agent::StackSpec & stack : side.slots ) {
            if ( !first ) {
                json += ',';
            }
            first = false;
            json += '[';
            json += std::to_string( stack.monsterId );
            json += ',';
            json += std::to_string( stack.count );
            json += ']';
        }
        json += ']';
        return json;
    }
}

std::string fheroes2::agent::computeDecisionDigest( const std::vector<DecisionRecord> & decisions )
{
    DigestWriter writer;
    writer.appendString( "agent_decisions_v0" );
    writer.appendU32( static_cast<uint32_t>( decisions.size() ) );

    for ( const DecisionRecord & decision : decisions ) {
        writer.appendU32( decision.engineDecisionIndex );
        writer.appendU32( decision.unitUid );
        writer.appendU32( static_cast<uint32_t>( decision.actions.size() ) );

        for ( const CommandSnapshot & action : decision.actions ) {
            writer.appendI32( static_cast<int32_t>( action.type ) );
            writer.appendU32( static_cast<uint32_t>( action.params.size() ) );
            for ( const int param : action.params ) {
                writer.appendI32( param );
            }
        }
    }

    return sha256Hex( writer.bytes() );
}

fheroes2::agent::TrajectoryWriter::TrajectoryWriter( const std::string & filePath )
    : _out( filePath, std::ios_base::out | std::ios_base::trunc )
{
    // A failed open is reported through isOpen(); the caller decides whether it is fatal.
}

void fheroes2::agent::TrajectoryWriter::writeHeader( const Scenario & scenario, const uint32_t mapSeed, const uint32_t combatSeed )
{
    _out << "{\"record\":\"episode_header\",\"schema\":\"agent_passive_v0\""
         << ",\"scenario_id\":\"" << escapeJson( scenario.scenarioId ) << '"' //
         << ",\"ground_type\":" << scenario.groundType //
         << ",\"tile_index\":" << scenario.tileIndex //
         << ",\"world_seed\":" << scenario.worldSeed //
         << ",\"max_rounds\":" << scenario.maxRounds //
         << ",\"attacker\":" << sideToJson( scenario.attacker ) //
         << ",\"defender\":" << sideToJson( scenario.defender ) //
         << ",\"map_seed\":" << mapSeed //
         << ",\"combat_seed\":" << combatSeed //
         << "}\n";
}

void fheroes2::agent::TrajectoryWriter::writeDecision( const DecisionRecord & decision )
{
    _out << "{\"record\":\"decision\",\"engine_decision_index\":" << decision.engineDecisionIndex //
         << ",\"unit_uid\":" << decision.unitUid //
         << ",\"actions\":[";

    bool first = true;
    for ( const CommandSnapshot & action : decision.actions ) {
        if ( !first ) {
            _out << ',';
        }
        first = false;

        _out << "{\"type\":\"" << commandTypeName( action.type ) << '"' //
             << ",\"key\":\"" << escapeJson( canonicalCommandKey( action ) ) << '"' //
             << ",\"params\":[";
        for ( size_t i = 0; i < action.params.size(); ++i ) {
            if ( i > 0 ) {
                _out << ',';
            }
            _out << action.params[i];
        }
        _out << "]}";
    }

    _out << "]}\n";
}

void fheroes2::agent::TrajectoryWriter::writeTerminal( const EpisodeOutcome & outcome, const size_t decisionCount, const std::string & decisionDigest )
{
    _out << "{\"record\":\"terminal\",\"termination\":\"" << terminationName( outcome.termination ) << '"' //
         << ",\"rounds\":" << outcome.rounds //
         << ",\"attacker_result\":" << outcome.attackerResult //
         << ",\"defender_result\":" << outcome.defenderResult //
         << ",\"decision_count\":" << decisionCount //
         << ",\"decision_digest\":\"" << escapeJson( decisionDigest ) << '"' //
         << ",\"state_digest\":\"" << escapeJson( outcome.stateDigest ) << '"' //
         << "}\n";
    _out.flush();
}
