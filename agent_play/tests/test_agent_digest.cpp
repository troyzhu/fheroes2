/***************************************************************************
 *   fheroes2 agent tests: DigestWriter and sha256Hex                       *
 *                                                                          *
 *   The SHA-256 implementation is self-contained (the engine has no hash   *
 *   facility), so it is pinned to the FIPS 180-4 test vectors here. The    *
 *   DigestWriter encoding is part of the digest contract: little-endian    *
 *   fixed-width integers and length-prefixed strings.                      *
 ***************************************************************************/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "agent_digest.h"

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

    std::vector<uint8_t> toBytes( const std::string & text )
    {
        return { text.begin(), text.end() };
    }
}

int main()
{
    using fheroes2::agent::DigestWriter;
    using fheroes2::agent::sha256Hex;

    std::printf( "test_agent_digest\n" );

    // FIPS 180-4 vectors.
    check( sha256Hex( {} ) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 of empty input" );
    check( sha256Hex( toBytes( "abc" ) ) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 of 'abc'" );
    check( sha256Hex( toBytes( "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" ) )
               == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
           "sha256 of the two-block standard vector" );

    {
        // One million 'a' characters: exercises many blocks and the length encoding.
        const std::vector<uint8_t> millionA( 1000000, static_cast<uint8_t>( 'a' ) );
        check( sha256Hex( millionA ) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "sha256 of one million 'a'" );
    }

    {
        // Exactly 55 and 56 bytes straddle the one-vs-two padding block boundary.
        const std::vector<uint8_t> b55( 55, static_cast<uint8_t>( 'x' ) );
        const std::vector<uint8_t> b56( 56, static_cast<uint8_t>( 'x' ) );
        check( sha256Hex( b55 ) != sha256Hex( b56 ), "padding boundary 55 vs 56 bytes differs" );
        check( sha256Hex( b55 ).size() == 64 && sha256Hex( b56 ).size() == 64, "digests are 64 hex characters" );
    }

    {
        DigestWriter writer;
        writer.appendU32( 0x01020304 );
        const std::vector<uint8_t> expected = { 0x04, 0x03, 0x02, 0x01 };
        check( writer.bytes() == expected, "appendU32 encodes little-endian" );
    }

    {
        DigestWriter writer;
        writer.appendI32( -1 );
        const std::vector<uint8_t> expected = { 0xFF, 0xFF, 0xFF, 0xFF };
        check( writer.bytes() == expected, "appendI32(-1) is two's complement" );
    }

    {
        DigestWriter writer;
        writer.appendU64( 0x1122334455667788ULL );
        const std::vector<uint8_t> expected = { 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11 };
        check( writer.bytes() == expected, "appendU64 encodes little-endian" );
    }

    {
        DigestWriter writer;
        writer.appendString( "ab" );
        const std::vector<uint8_t> expected = { 0x02, 0x00, 0x00, 0x00, 'a', 'b' };
        check( writer.bytes() == expected, "appendString is length-prefixed" );
    }

    {
        // Field aliasing must be impossible: ("a" + "bc") and ("ab" + "c") differ.
        DigestWriter first;
        first.appendString( "a" );
        first.appendString( "bc" );
        DigestWriter second;
        second.appendString( "ab" );
        second.appendString( "c" );
        check( sha256Hex( first.bytes() ) != sha256Hex( second.bytes() ), "length prefixes prevent field aliasing" );
    }

    std::printf( "%d passed, %d failed\n", passed, failed );
    return ( failed == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE;
}
