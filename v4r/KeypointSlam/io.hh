/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_ARTICULATED_OBJECT_IO_HH
#define KP_ARTICULATED_OBJECT_IO_HH

#include <iostream>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include "ArticulatedObject.hh"
#include "ArticulatedObject_serialization.hpp"


namespace kp
{



/*************************************************************************** 
 * io 
 */
class io
{
public:
  io() {};

  /** write **/
  static void write(const std::string &file, const ArticulatedObject::Ptr &model);

  /** read **/
  static bool read(const std::string &file, ArticulatedObject::Ptr &model);
};





} //--END--

#endif

