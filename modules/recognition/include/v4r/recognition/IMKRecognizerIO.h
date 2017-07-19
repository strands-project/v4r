/******************************************************************************
 * Copyright (c) 2016 Johann Prankl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/


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
  static void write(const std::string &dir, const std::vector<std::string> &object_names, const std::vector<IMKView> &object_models, const CodebookMatcher &cb, const std::string &codebookFilename="");

  /** read **/
  static bool read(const std::string &dir, std::vector<std::string> &object_names, std::vector<IMKView> &object_models, CodebookMatcher &cb, const std::string &codebookFilename="");
};





} //--END--

#endif

