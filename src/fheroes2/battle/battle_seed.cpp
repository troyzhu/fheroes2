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

#include "battle_seed.h"

#include <cstddef>

#include "army.h"
#include "army_troop.h"
#include "rand.h"

uint32_t Battle::computeBattleSeed( const int32_t mapIndex, const uint32_t mapSeed, const Army & attackingArmy, const Army & defendingArmy )
{
    uint32_t seed = static_cast<uint32_t>( mapIndex ) + mapSeed;

    for ( size_t i = 0; i < attackingArmy.Size(); ++i ) {
        const Troop * troop = attackingArmy.GetTroop( i );
        if ( troop->isValid() ) {
            Rand::combineSeedWithValueHash( seed, troop->GetID() );
            Rand::combineSeedWithValueHash( seed, troop->GetCount() );
        }
        else {
            Rand::combineSeedWithValueHash( seed, 0 );
        }
    }

    for ( size_t i = 0; i < defendingArmy.Size(); ++i ) {
        const Troop * troop = defendingArmy.GetTroop( i );
        if ( troop->isValid() ) {
            Rand::combineSeedWithValueHash( seed, troop->GetID() );
            Rand::combineSeedWithValueHash( seed, troop->GetCount() );
        }
        else {
            Rand::combineSeedWithValueHash( seed, 0 );
        }
    }

    return seed;
}
