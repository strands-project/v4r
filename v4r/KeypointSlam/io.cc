/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "io.hh"

namespace kp
{


/** 
 * write 
 */
void io::write(const std::string &file, const ArticulatedObject::Ptr &model)
{
  std::ofstream ofs(file.c_str());
  boost::archive::binary_oarchive oa(ofs);
  oa << model;
}

/** 
 * read 
 */
bool io::read(const std::string &file, ArticulatedObject::Ptr &model)
{
  std::ifstream ifs(file.c_str());
  if (ifs.is_open())
  {
    boost::archive::binary_iarchive ia(ifs);
    ia >> model;
    return true;
  }
  return false;
}



} //--END--


