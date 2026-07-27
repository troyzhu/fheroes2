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

#include "agent_digest.h"

#include <array>
#include <cstddef>

namespace
{
    constexpr std::array<uint32_t, 64> sha256RoundConstants = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };

    uint32_t rotateRight( const uint32_t value, const uint32_t bits )
    {
        return ( value >> bits ) | ( value << ( 32 - bits ) );
    }

    void processBlock( std::array<uint32_t, 8> & state, const uint8_t * block )
    {
        std::array<uint32_t, 64> w{};

        for ( size_t i = 0; i < 16; ++i ) {
            w[i] = ( static_cast<uint32_t>( block[i * 4] ) << 24 ) | ( static_cast<uint32_t>( block[i * 4 + 1] ) << 16 )
                   | ( static_cast<uint32_t>( block[i * 4 + 2] ) << 8 ) | static_cast<uint32_t>( block[i * 4 + 3] );
        }
        for ( size_t i = 16; i < 64; ++i ) {
            const uint32_t s0 = rotateRight( w[i - 15], 7 ) ^ rotateRight( w[i - 15], 18 ) ^ ( w[i - 15] >> 3 );
            const uint32_t s1 = rotateRight( w[i - 2], 17 ) ^ rotateRight( w[i - 2], 19 ) ^ ( w[i - 2] >> 10 );
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];

        for ( size_t i = 0; i < 64; ++i ) {
            const uint32_t bigS1 = rotateRight( e, 6 ) ^ rotateRight( e, 11 ) ^ rotateRight( e, 25 );
            const uint32_t choice = ( e & f ) ^ ( ~e & g );
            const uint32_t temp1 = h + bigS1 + choice + sha256RoundConstants[i] + w[i];
            const uint32_t bigS0 = rotateRight( a, 2 ) ^ rotateRight( a, 13 ) ^ rotateRight( a, 22 );
            const uint32_t majority = ( a & b ) ^ ( a & c ) ^ ( b & c );
            const uint32_t temp2 = bigS0 + majority;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
}

std::string fheroes2::agent::sha256Hex( const std::vector<uint8_t> & data )
{
    std::array<uint32_t, 8> state = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

    const size_t fullBlocks = data.size() / 64;
    for ( size_t i = 0; i < fullBlocks; ++i ) {
        processBlock( state, data.data() + i * 64 );
    }

    // Padding: 0x80, zeros, then the message length in bits as a 64-bit big-endian integer.
    std::array<uint8_t, 128> tail{};
    const size_t remaining = data.size() - fullBlocks * 64;
    for ( size_t i = 0; i < remaining; ++i ) {
        tail[i] = data[fullBlocks * 64 + i];
    }
    tail[remaining] = 0x80;

    const size_t tailBlocks = ( remaining + 1 + 8 > 64 ) ? 2 : 1;
    const uint64_t bitLength = static_cast<uint64_t>( data.size() ) * 8;
    for ( size_t i = 0; i < 8; ++i ) {
        tail[tailBlocks * 64 - 1 - i] = static_cast<uint8_t>( ( bitLength >> ( i * 8 ) ) & 0xFF );
    }

    for ( size_t i = 0; i < tailBlocks; ++i ) {
        processBlock( state, tail.data() + i * 64 );
    }

    static const char * hexDigits = "0123456789abcdef";
    std::string result;
    result.reserve( 64 );
    for ( const uint32_t word : state ) {
        for ( int shift = 28; shift >= 0; shift -= 4 ) {
            result.push_back( hexDigits[( word >> shift ) & 0xF] );
        }
    }

    return result;
}
