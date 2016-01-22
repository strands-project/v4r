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

#ifndef V4R_CODEBOOK_MATCHER_HH
#define V4R_CODEBOOK_MATCHER_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/common/ClusteringRNN.h>
#include <v4r/core/macros.h>
#include <v4r/keypoints/impl/triple.hpp>

namespace v4r 
{

class V4R_EXPORTS CodebookMatcher
{
public:
  class V4R_EXPORTS Parameter
  {
  public:
    float thr_desc_rnn;
    float nnr;
    float max_dist;
    Parameter(float _thr_desc_rnn=0.55, float _nnr=0.92, float _max_dist=.7)
    : thr_desc_rnn(_thr_desc_rnn), nnr(_nnr), max_dist(_max_dist) {}
  };

private:
  Parameter param;

  ClusteringRNN rnn;

  int max_view_index;
  DataMatrix2Df descs;
  std::vector< std::pair<int,int> > vk_indices;

  cv::Mat cb_centers;
  std::vector< std::vector< std::pair<int,int> > > cb_entries;
  std::vector< std::pair<int, int> > view_rank;

  cv::Ptr<cv::DescriptorMatcher> matcher;

public:
  cv::Mat dbg;

  CodebookMatcher(const Parameter &p=Parameter());
  ~CodebookMatcher();

  void clear();
  void addView(const cv::Mat &descriptors, int view_idx);
  void createCodebook();
  void createCodebook(cv::Mat &_cb_centers, std::vector< std::vector< std::pair<int,int> > > &_cb_entries);
  void setCodebook(const cv::Mat &_cb_centers, const std::vector< std::vector< std::pair<int,int> > > &_cb_entries);
  void queryViewRank(const cv::Mat &descriptors, std::vector< std::pair<int, int> > &view_rank);
  void queryMatches(const cv::Mat &descriptors, std::vector< std::vector< cv::DMatch > > &matches);

  inline const std::vector< std::vector< std::pair<int,int> > > &getEntries() { return cb_entries; }
  inline const cv::Mat &getDescriptors() { return cb_centers; }
  inline const std::vector< std::pair<int, int> > &getViewRank() {return view_rank;}

  typedef SmartPtr< ::v4r::CodebookMatcher> Ptr;
  typedef SmartPtr< ::v4r::CodebookMatcher const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

