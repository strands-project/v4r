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

#include <v4r/keypoints/CodebookMatcher.h>
#include <v4r/common/impl/ScopeTime.hpp>


namespace v4r
{


using namespace std;


inline bool cmpViewRandDec(const std::pair<int,int> &i, const std::pair<int,int> &j)
{
  return (i.second>j.second);
}


/************************************************************************************
 * Constructor/Destructor
 */
CodebookMatcher::CodebookMatcher(const Parameter &p)
 : param(p), max_view_index(0)
{ 
  rnn.dbg = true;
}

CodebookMatcher::~CodebookMatcher()
{
}





/***************************************************************************************/

/**
 * @brief CodebookMatcher::clear
 */
void CodebookMatcher::clear()
{
  max_view_index = 0;
  descs.clear();
  vk_indices.clear();
}

/**
 * @brief CodebookMatcher::addView
 * @param descriptors
 * @param view_idx
 */
void CodebookMatcher::addView(const cv::Mat &descriptors, int view_idx)
{
  descs.reserve(descs.rows+descriptors.rows, descriptors.cols);
  vk_indices.reserve(vk_indices.size()+descriptors.rows);

  for (unsigned i=0; i<(unsigned)descriptors.rows; i++)
  {
    descs.push_back(&descriptors.at<float>(i,0), descriptors.cols);
    vk_indices.push_back(std::make_pair(view_idx,i));
  }

  if (view_idx > max_view_index)
    max_view_index = view_idx;
}

/**
 * @brief CodebookMatcher::createCodebook
 */
void CodebookMatcher::createCodebook()
{
  v4r::ScopeTime t("CodebookMatcher::createCodebook");

  // rnn clustering
  v4r::DataMatrix2Df centers;
  std::vector<std::vector<int> > clusters;

  rnn.param.dist_thr = param.thr_desc_rnn;
  rnn.cluster(descs);
  rnn.getClusters(clusters);
  rnn.getCenters(centers);

  cb_entries.clear();
  cb_entries.resize(clusters.size());
  cb_centers = cv::Mat_<float>(clusters.size(), centers.cols);

  for (unsigned i=0; i<clusters.size(); i++)
  {
    for (unsigned j=0; j<clusters[i].size(); j++)
      cb_entries[i].push_back(vk_indices[clusters[i][j]]);

    cv::Mat_<float>(1,centers.cols,&centers(i,0)).copyTo(cb_centers.row(i));
  }

  cout<<"codbeook.size()="<<clusters.size()<<"/"<<descs.rows<<endl;

  // create flann for matching
  { v4r::ScopeTime t("create FLANN");
  matcher = new cv::FlannBasedMatcher();
  matcher->add(std::vector<cv::Mat>(1,cb_centers));
  matcher->train();
  }

  // once the codebook is created clear the temp containers
  rnn = ClusteringRNN();
  descs = DataMatrix2Df();
  vk_indices = std::vector< std::pair<int,int> >();
  cb_centers.release();

}

/**
 * @brief CodebookMatcher::createCodebook
 * @param _cb_centers
 * @param _cb_entries
 */
void CodebookMatcher::createCodebook(cv::Mat &_cb_centers, std::vector< std::vector< std::pair<int,int> > > &_cb_entries)
{
  v4r::ScopeTime t("CodebookMatcher::createCodebook");

  // rnn clustering
  v4r::DataMatrix2Df centers;
  std::vector<std::vector<int> > clusters;

  rnn.param.dist_thr = param.thr_desc_rnn;
  rnn.cluster(descs);
  rnn.getClusters(clusters);
  rnn.getCenters(centers);

  cb_entries.clear();
  cb_entries.resize(clusters.size());
  cb_centers = cv::Mat_<float>(clusters.size(), centers.cols);

  for (unsigned i=0; i<clusters.size(); i++)
  {
    for (unsigned j=0; j<clusters[i].size(); j++)
      cb_entries[i].push_back(vk_indices[clusters[i][j]]);

    cv::Mat_<float>(1,centers.cols,&centers(i,0)).copyTo(cb_centers.row(i));
  }

  cout<<"codbeook.size()="<<clusters.size()<<"/"<<descs.rows<<endl;

  // create flann for matching
  { v4r::ScopeTime t("create FLANN");
  matcher = new cv::FlannBasedMatcher();
  matcher->add(std::vector<cv::Mat>(1,cb_centers));
  matcher->train();
  }

  // return codebook
  cb_centers.copyTo(_cb_centers);
  _cb_entries = cb_entries;

  // once the codebook is created clear the temp containers
  rnn = ClusteringRNN();
  descs = DataMatrix2Df();
  vk_indices = std::vector< std::pair<int,int> >();
  cb_centers.release();
}

/**
 * @brief CodebookMatcher::setCodebook
 * @param _cb_centers
 * @param _cb_entries
 */
void CodebookMatcher::setCodebook(const cv::Mat &_cb_centers, const std::vector< std::vector< std::pair<int,int> > > &_cb_entries)
{
  cb_centers = _cb_centers;
  cb_entries = _cb_entries;

  max_view_index = 0;

  for (unsigned i=0; i<cb_entries.size(); i++)
    for (unsigned j=0; j<cb_entries[i].size(); j++)
      if (cb_entries[i][j].first > max_view_index)
        max_view_index = cb_entries[i][j].first;

  max_view_index++;

  // create flann for matching
  { v4r::ScopeTime t("create FLANN");
  matcher = new cv::FlannBasedMatcher();
  matcher->add(std::vector<cv::Mat>(1,cb_centers));
  matcher->train();
  }


}

/**
 * @brief CodebookMatcher::queryViewRank
 * @param descriptors
 * @param view_rank <view_index, rank_number>  sorted better first
 */
void CodebookMatcher::queryViewRank(const cv::Mat &descriptors, std::vector< std::pair<int, int> > &view_rank)
{
  std::vector< std::vector< cv::DMatch > > cb_matches;

  matcher->knnMatch( descriptors, cb_matches, 2 );

  view_rank.resize(max_view_index+1);

  for (unsigned i=0; i<view_rank.size(); i++)
    view_rank[i] = std::make_pair((int)i,0.);

  for (unsigned i=0; i<cb_matches.size(); i++)
  {
    if (cb_matches[i].size()>1)
    {
      cv::DMatch &ma0 = cb_matches[i][0];

      if (ma0.distance/cb_matches[i][1].distance < param.nnr)
      {
        const std::vector< std::pair<int,int> > &occs = cb_entries[ma0.trainIdx];

        for (unsigned j=0; j<occs.size(); j++)
          view_rank[occs[j].first].second++;
      }
    }
  }

  //sort
  std::sort(view_rank.begin(),view_rank.end(),cmpViewRandDec);
}

/**
 * @brief CodebookMatcher::queryMatches
 * @param descriptors
 * @param matches
 */
void CodebookMatcher::queryMatches(const cv::Mat &descriptors, std::vector< std::vector< cv::DMatch > > &matches)
{
  std::vector< std::vector< cv::DMatch > > cb_matches;

  matcher->knnMatch( descriptors, cb_matches, 2 );

  matches.clear();
  matches.resize(descriptors.rows);
  view_rank.resize(max_view_index+1);

  for (unsigned i=0; i<view_rank.size(); i++)
    view_rank[i] = std::make_pair((int)i,0.);

  for (unsigned i=0; i<cb_matches.size(); i++)
  {
    if (cb_matches[i].size()>1)
    {
      cv::DMatch &ma0 = cb_matches[i][0];
      if (ma0.distance < param.max_dist && ma0.distance/cb_matches[i][1].distance < param.nnr)
      {
        std::vector< cv::DMatch > &ms = matches[ma0.queryIdx];
        const std::vector< std::pair<int,int> > &occs = cb_entries[ma0.trainIdx];

        for (unsigned j=0; j<occs.size(); j++)
        {
          ms.push_back(cv::DMatch(ma0.queryIdx,occs[j].second,occs[j].first,ma0.distance));
          view_rank[occs[j].first].second++;
        }
      }
    }
  }

  //sort
  std::sort(view_rank.begin(),view_rank.end(),cmpViewRandDec);
}




}












