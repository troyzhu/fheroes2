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

class Army;

namespace Battle
{
    // Computes the seed for the combat random generator: fold the battle tile index and the
    // world map seed with every army slot (monster id and count for valid troops, a single
    // zero for empty slots), attacker army first, defender army second. The traversal runs
    // over Army::Size() rather than a hardcoded slot count.
    //
    // The engine battle loader and the agent battle environment must derive their combat
    // seeds through this one function so that identical inputs replay identical battles in
    // both. Any change to the folding order or contents breaks recorded-trajectory
    // compatibility.
    uint32_t computeBattleSeed( const int32_t mapIndex, const uint32_t mapSeed, const Army & attackingArmy, const Army & defendingArmy );
}
