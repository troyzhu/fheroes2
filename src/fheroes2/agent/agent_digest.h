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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fheroes2::agent
{
    // Order-sensitive byte accumulator used to build the canonical representation that agent
    // state digests are computed over (agent spec, section 12.5). Integers are encoded with a
    // fixed width in little-endian order so the digest is identical across platforms and build
    // types; strings are length-prefixed so that concatenated fields can never alias each other.
    class DigestWriter
    {
    public:
        void appendU8( const uint8_t value )
        {
            _bytes.push_back( value );
        }

        void appendU32( const uint32_t value )
        {
            for ( int i = 0; i < 4; ++i ) {
                _bytes.push_back( static_cast<uint8_t>( ( value >> ( i * 8 ) ) & 0xFF ) );
            }
        }

        void appendI32( const int32_t value )
        {
            // Two's complement representation of the value, encoded like a u32.
            appendU32( static_cast<uint32_t>( value ) );
        }

        void appendU64( const uint64_t value )
        {
            for ( int i = 0; i < 8; ++i ) {
                _bytes.push_back( static_cast<uint8_t>( ( value >> ( i * 8 ) ) & 0xFF ) );
            }
        }

        void appendString( const std::string & value )
        {
            appendU32( static_cast<uint32_t>( value.size() ) );
            for ( const char ch : value ) {
                _bytes.push_back( static_cast<uint8_t>( ch ) );
            }
        }

        const std::vector<uint8_t> & bytes() const
        {
            return _bytes;
        }

    private:
        std::vector<uint8_t> _bytes;
    };

    // SHA-256 (FIPS 180-4) of the given bytes, as a 64-character lowercase hex string. The
    // repository ships no general-purpose hash facility, so the implementation is self-contained;
    // it is validated against the standard test vectors in agent_play/tests/test_agent_digest.cpp.
    std::string sha256Hex( const std::vector<uint8_t> & data );
}
