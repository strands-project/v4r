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

#include <v4r/features/FeatureSelection.h>
#include <v4r/common/impl/ScopeTime.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace v4r
{
using namespace std;

/************************************************************************************
 * Constructor/Destructor
 */
FeatureSelection::FeatureSelection(const Parameter &p)
 : param(p)
{ 
  rnn.dbg = false;
}

FeatureSelection::~FeatureSelection()
{
}


/***************************************************************************************/

/**
 * compute
 */
void FeatureSelection::compute(std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
//ScopeTime t("FeatureSelection::compute");
  rnn.param.dist_thr = param.thr_image_px;
  std::vector<std::vector<int> > pts_clusters, desc_clusters;
  std::vector<cv::KeyPoint> tmp_keys;
  cv::Mat tmp_descs = cv::Mat_<float>(0,descriptors.cols);
  v4r::DataMatrix2Df centers;

  // cluster keypoints
  pts.clear();
  pts.reserve(keys.size(),2);
  
  for (unsigned i=0; i<keys.size(); i++)
    pts.push_back(&keys[i].pt.x, 2);

  rnn.cluster(pts);
  rnn.getClusters(pts_clusters);

  // cluster descriptors
  rnn.param.dist_thr = param.thr_desc;

//unsigned cnt=0;
//cv::Mat tmp;
//dbg.copyTo(tmp);
  for (unsigned i=0; i<pts_clusters.size(); i++)
  {
    if (pts_clusters[i].size() == 0) continue;

    if (pts_clusters[i].size() == 1)
    {
      tmp_descs.push_back(descriptors.row(pts_clusters[i][0]));
      tmp_keys.push_back(keys[pts_clusters[i][0]]);
//cv::KeyPoint &key = tmp_keys.back();
//cv::circle( tmp, key.pt, key.size/2, cv::Scalar(255), 1, CV_AA );
      continue;
    }

    descs.clear();
    descs.reserve(pts_clusters[i].size(), descriptors.cols);

    for (unsigned j=0; j<pts_clusters[i].size(); j++)
      descs.push_back(&descriptors.at<float>(pts_clusters[i][j],0), descriptors.cols);

    rnn.cluster(descs);
    rnn.getClusters(desc_clusters);
    rnn.getCenters(centers);

    for (unsigned j=0; j<desc_clusters.size(); j++)
    {
      unsigned size = desc_clusters[j].size();
      float inv_size = 1./(float)size;
      tmp_descs.push_back(cv::Mat_<float>(1,descriptors.cols,&centers(j,0)));
      tmp_keys.push_back(cv::KeyPoint(0,0,0,0,0,0,-1));
      cv::KeyPoint &mkey = tmp_keys.back();

      for (unsigned k=0; k<size; k++)
      {
        cv::KeyPoint &key = keys[pts_clusters[i][desc_clusters[j][k]]];
        mkey.pt += key.pt;
        mkey.size += key.size;
        mkey.angle += key.angle;
        mkey.response += key.response;
        mkey.octave += key.octave;
//cv::circle( tmp, key.pt, key.size/2, cv::Scalar(0), 1, CV_AA );
      }

      mkey.pt *= inv_size;
      mkey.size *= inv_size;
      mkey.angle *= inv_size;
      mkey.response *= inv_size;
      mkey.octave *= inv_size;
//cv::KeyPoint &key = tmp_keys.back();
//cv::circle( tmp, key.pt, key.size/2, cv::Scalar(255), 1, CV_AA );
    }

//    cout<<"desc_clusters.size()="<<desc_clusters.size();
//for (unsigned j=0; j<desc_clusters.size(); j++)
//{
//  cv::Mat tmp;
//  dbg.copyTo(tmp);

//  for (unsigned k=0; k<desc_clusters[j].size(); k++)
//  {
//    cv::KeyPoint &key = keys[pts_clusters[i][desc_clusters[j][k]]];
//    cv::circle( tmp, key.pt, key.size/2, cv::Scalar(255), 1, CV_AA );
//  }
//if (desc_clusters[j].size()>=2) {
//  char filename[PATH_MAX];
//  snprintf(filename,PATH_MAX,"log/dbg_%04d.png",cnt);
//  cv::imwrite(filename,tmp);
//  cnt++;
//  cout<<desc_clusters[j].size()<<" ";
//  cout<<" | ";
//}
//}
//cout<<endl;

//cout<<"--"<<endl;
  }

//cv::imwrite("log/dbg.png",tmp);


  tmp_descs.copyTo(descriptors);
  keys = tmp_keys;

  // mark near by points
  /*pts.clear();
  pts.reserve(keys.size(),2);

  for (unsigned i=0; i<keys.size(); i++)
    pts.push_back(&keys[i].pt.x, 2);

  rnn.cluster(pts);
  rnn.getClusters(pts_clusters);

  for (unsigned i=0; i<pts_clusters.size(); i++)
  {
    for (unsigned j=1; j<pts_clusters[i].size(); j++)
      keys[pts_clusters[i][j]].response = -1;
  }*/
}
}
