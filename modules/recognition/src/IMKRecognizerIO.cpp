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


#include <v4r/recognition/IMKRecognizerIO.h>
#include <boost/algorithm/string.hpp>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <v4r/io/filesystem.h>




namespace v4r
{


using namespace std;



/************************ PRIVATE ****************************/

/**
 * @brief IMKRecognizerIO::generateDir
 * @param dir
 * @param object_names
 * @param full_dir
 */
void IMKRecognizerIO::generateDir(const std::string &dir, const std::vector<std::string> &object_names, std::string &full_dir)
{
  std::vector<std::string> sorted_names = object_names;
  std::sort(sorted_names.begin(),sorted_names.end());
  full_dir = dir+std::string("/IMK_");

  for (int i=0; i<(int)sorted_names.size(); i++)
  {
    full_dir+=sorted_names[i];
    if (i<((int)sorted_names.size())-1) full_dir+=std::string("_");
  }
}

/**
 * @brief IMKRecognizerIO::generateName
 * @param dir
 * @param object_names
 * @param full_name
 */
void IMKRecognizerIO::generateName(const std::string &dir, const std::vector<std::string> &object_names, std::string &full_name)
{
  std::vector<std::string> sorted_names = object_names;
  std::sort(sorted_names.begin(),sorted_names.end());
  full_name = dir+std::string("/imk_");
  for (int i=0; i<(int)sorted_names.size(); i++)
  {
    full_name+=sorted_names[i];
    if (i<((int)sorted_names.size())-1) full_name+=std::string("_");
  }
  full_name+=std::string(".bin");
}

/************************ PUBLIC *****************************/

/** 
 * write 
 */
void IMKRecognizerIO::write(const std::string &dir, const std::vector<std::string> &object_names, const std::vector<IMKView> &object_models, const CodebookMatcher &cb, const std::string &codebookFilename)
{
//  std::string full_dir;
//  generateDir(dir, object_names, full_dir);
//  boost::filesystem::create_directories(full_dir);

  std::string full_name;
  if (!codebookFilename.empty()) {
     full_name = dir + "/" + codebookFilename;
  } else {
    generateName(dir, object_names, full_name);
  }

  cv::Mat cb_centers = cb.getDescriptors();
  std::vector< std::vector< std::pair<int,int> > > cb_entries = cb.getEntries();
//  std::ofstream ofs((full_dir+std::string("/imk_recognizer_model.bin")).c_str());
  std::ofstream ofs(full_name.c_str());

  boost::archive::binary_oarchive oa(ofs);
  oa << object_names;
  oa << object_models;
  oa << cb_centers;
  oa << cb_entries;
}

/** 
 * read 
 */
bool IMKRecognizerIO::read(const std::string &dir, std::vector<std::string> &object_names, std::vector<IMKView> &object_models, CodebookMatcher &cb, const std::string &codebookFilename)
{
//  std::string full_dir;
//  generateDir(dir, object_names, full_dir);
//  std::ifstream ifs((full_dir+std::string("/imk_recognizer_model.bin")).c_str());
  std::string full_name;
  if (!codebookFilename.empty()) {
     full_name = dir + "/" + codebookFilename;
  } else {
    generateName(dir, object_names, full_name);
  }
  std::ifstream ifs(full_name.c_str());
  if (ifs.is_open())
  {
//    cout<<(full_dir+std::string("/imk_recognizer_model.bin"))<<endl;
    cout<<full_name<<endl;
    cv::Mat cb_centers;
    std::vector< std::vector< std::pair<int,int> > > cb_entries;
    boost::archive::binary_iarchive ia(ifs);
    ia >> object_names;
    ia >> object_models;
    ia >> cb_centers;
    ia >> cb_entries;
    cb.setCodebook( cb_centers, cb_entries );
    return true;
  }
  return false;
}



} //--END--


