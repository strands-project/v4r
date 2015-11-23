/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#ifndef V4R_DATACONTAINER_HH
#define V4R_DATACONTAINER_HH

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string.h>
#include <v4r/common/impl/SmartPtr.hpp>

namespace v4r 
{

/**
 * DataContainer
 */
class V4R_EXPORTS DataContainer
{
public:
  enum Type
  {
    FLOAT_2D,
    FLOAT3_2D,
    FLOAT4_2D,
    UCHAR_2D,
    INT_2D,
    UINT_2D,
    MAX_TYPE,
    UNDEF = MAX_TYPE
  };

public:
  Type type;

  DataContainer(const Type &_type=UNDEF) {type=_type;};
  virtual ~DataContainer(){};

  typedef SmartPtr< ::v4r::DataContainer > Ptr;
  typedef SmartPtr< ::v4r::DataContainer const> ConstPtr;
};


} //--END--

#endif

