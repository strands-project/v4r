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


#include <v4r/recognition/IMKObjectVotesClustering.h>
#include <v4r/common/impl/Vector.hpp>
#include <v4r/keypoints/impl/triple.hpp>

#include "opencv2/highgui/highgui.hpp"
//#include "v4r/KeypointTools/ScopeTime.hpp"

//#define DEBUG_VOTE_CL


namespace v4r
{

using namespace std;

const float IMKObjectVotesClustering::two_pi = 2.*M_PI;



inline bool cmpCluster(const boost::shared_ptr<v4r::triple<unsigned, double, std::vector< cv::DMatch > > > &a, const boost::shared_ptr<v4r::triple<unsigned, double, std::vector< cv::DMatch > > > &b)
{
  return (a->second>b->second);
}





/************************************************************************************
 * Constructor/Destructor
 */
IMKObjectVotesClustering::IMKObjectVotesClustering(const Parameter &p)
 : param(p)
{
}

IMKObjectVotesClustering::~IMKObjectVotesClustering()
{
}

/**
 * @brief IMKObjectVotesClustering::createVotes
 * @param id
 * @param views
 * @param keys
 * @param matches
 * @param votes
 * @param voting_matches
 */
void IMKObjectVotesClustering::createVotes(unsigned id, const std::vector<IMKView> &views, const std::vector<cv::KeyPoint> &keys, const std::vector< std::vector< cv::DMatch > > &matches, DataMatrix2Df &votes, std::vector< cv::DMatch > &voting_matches, std::vector<double> &weights)
{
  float scale, delta_angle;
  cv::Point2f pt, pt_scaled;

  votes.resize(0,2);
  voting_matches.clear();
  weights.clear();


#ifdef DEBUG_VOTE_CL
  cv::Mat im_tmp;
  if (!dbg.empty())
    dbg.copyTo(im_tmp);
#endif

  for (unsigned i=0; i<matches.size(); i++)
  {
    const std::vector< cv::DMatch > &ms = matches[i];
    for (unsigned j=0; j<ms.size(); j++)
    {
      const cv::DMatch &m = ms[j];
      if (views[m.imgIdx].object_id == id)
      {
        const cv::KeyPoint &query_key = keys[m.queryIdx];
        const cv::KeyPoint &train_key = views[m.imgIdx].keys[m.trainIdx];

        scale = query_key.size/train_key.size;
        delta_angle = diffAngle_0_2pi(query_key.angle*M_PI/180., train_key.angle*M_PI/180.);

        pt_scaled = scale*train_key.pt;
        rotate2(&pt_scaled.x, delta_angle, &pt.x);
        pt = pt+query_key.pt;

//    pt = train_key.pt+query_key.pt;

        weights.push_back(1./(double)ms.size());
        voting_matches.push_back(m);
        votes.push_back(&pt.x, 2);

#ifdef DEBUG_VOTE_CL
  if (!dbg.empty())
  {
    cv::line(im_tmp, query_key.pt, pt, CV_RGB(255,255,255), 1);
    cv::circle(im_tmp, pt, 2, CV_RGB(255,0,0),2);
  }
#endif

      }
    }
  }

#ifdef DEBUG_VOTE_CL
  if (!dbg.empty())
  {
    cv::imshow("dbg votes", im_tmp);
    cv::waitKey(0);
  }
#endif
}



/******************************* PUBLIC ***************************************/

/**
 * @brief IMKObjectVotesClustering::operate
 * @param object_names
 * @param views
 * @param keys
 * @param matches
 * @param clusters DMatch::distance .. 1./number of occurances
 */
void IMKObjectVotesClustering::operate(const std::vector<std::string> &object_names, const std::vector<IMKView> &views, const std::vector<cv::KeyPoint> &keys, const std::vector< std::vector< cv::DMatch > > &matches, std::vector< boost::shared_ptr< v4r::triple<unsigned, double, std::vector< cv::DMatch > > > > &clusters)
{
  rnn.param.dist_thr = param.cluster_dist;
  rnn.dbg = false;
  clusters.clear();

  for (unsigned i=0; i<object_names.size(); i++)
  {
    createVotes(i, views, keys, matches, votes, voting_matches, weights);

    rnn.cluster(votes);
    rnn.getClusters(indices);

    for (unsigned j=0; j<indices.size(); j++)
    {
      const std::vector<int> &inds = indices[j];
      clusters.push_back( boost::shared_ptr< v4r::triple<unsigned, double, std::vector< cv::DMatch > > >(new v4r::triple<unsigned, double, std::vector< cv::DMatch > >()) );
      v4r::triple<unsigned, double, std::vector< cv::DMatch > > &cl = *clusters.back();
      cl.first = i;
      cl.second = 0.;
      for (unsigned k=0; k<inds.size(); k++)
      {
        cl.third.push_back(voting_matches[inds[k]]);
        cl.third.back().distance = weights[inds[k]];
        cl.second += weights[inds[k]];
      }
//      if (cl.second>10) cout<<inds.size()<<"("<<cl.second<<") ";
    }
  }
//  cout<<endl;

  // sort clusters
  std::sort(clusters.begin(), clusters.end(), cmpCluster);


#ifdef DEBUG_VOTE_CL
  if (!dbg.empty())
  {
    cv::Mat im_tmp;
    float scale, delta_angle;
    cv::Point2f pt, pt_scaled;

    for (unsigned i=0; i<clusters.size() && i<50; i++)
    {
      dbg.copyTo(im_tmp);
      const v4r::triple<unsigned, double, std::vector< cv::DMatch > > &cl = *clusters[i];
      cv::Vec3b col(rand()%255,rand()%255,rand()%255);

      cout<<i<<": obj="<<cl.first<<", weight="<<cl.second<<endl;
      if (cl.second<4) break;

      for (unsigned j=0; j<cl.third.size(); j++)
      {
        const cv::DMatch &m = cl.third[j];
        const cv::KeyPoint &query_key = keys[m.queryIdx];
        const cv::KeyPoint &train_key = views[m.imgIdx].keys[m.trainIdx];

        scale = query_key.size/train_key.size;
        delta_angle = diffAngle_0_2pi(query_key.angle*M_PI/180., train_key.angle*M_PI/180.);

        pt_scaled = scale*train_key.pt;
        rotate2(&pt_scaled.x, delta_angle, &pt.x);
        pt = pt+query_key.pt;

        cv::line(im_tmp, pt, query_key.pt, CV_RGB(255,255,255), 1);
        cv::circle(im_tmp, pt, 2, CV_RGB(col[0],col[1],col[2]),2);
      }

      cv::imshow("dbg votes", im_tmp);
      cv::waitKey(0);
    }

    cout<<"-- end --"<<endl;
    cv::waitKey(0);
  }
#endif
}




}












