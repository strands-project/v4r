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

#ifndef KP_IMK_RECOGNIZER_IO_HH
#define KP_IMK_RECOGNIZER_IO_HH

#include <iostream>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <v4r/recognition/IMKRecognizer.h>
#include <v4r/recognition/IMKRecognizer_serialization.hpp>
#include <v4r/keypoints/CodebookMatcher.h>
#include <opencv2/core/core.hpp>
#include "boost/filesystem.hpp"


namespace v4r
{



/*************************************************************************** 
 * IMKRecognizerIO
 */
class V4R_EXPORTS IMKRecognizerIO
{
private:
  static void generateDir(const std::string &dir, const std::vector<std::string> &object_names, std::string &full_dir);
  static void generateName(const std::string &dir, const std::vector<std::string> &object_names, std::string &full_name);

public:
  IMKRecognizerIO() {};

  /** write **/
  static void write(const std::string &dir, const std::vector<std::string> &object_names, const std::vector<IMKView> &object_models, const CodebookMatcher &cb);

  /** read **/
  static bool read(const std::string &dir, std::vector<std::string> &object_names, std::vector<IMKView> &object_models, CodebookMatcher &cb);
};





} //--END--

#endif

