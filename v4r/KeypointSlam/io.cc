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
bool io::write(const std::string &file, const ArticulatedObject::Ptr &model)
{
  if (model.get()==0)
    return false;

  std::ofstream ofs(file.c_str());
  boost::archive::binary_oarchive oa(ofs);
  oa << model;

  return true;
}

/** 
 * read 
 */
bool io::read(const std::string &file, ArticulatedObject::Ptr &model)
{
  model.reset(new ArticulatedObject());

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


