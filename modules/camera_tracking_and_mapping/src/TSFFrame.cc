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
 * @author Johann Prankl
 *
 */


#include <v4r/camera_tracking_and_mapping/TSFFrame.hh>




namespace v4r
{


using namespace std;




/************************************************************************************
 * Constructor/Destructor
 */
TSFFrame::TSFFrame()
  : idx(-1), delta_cloud_rgb_pose(Eigen::Matrix4f::Identity()), fw_link(-1), bw_link(-1), have_track(false)
{
}

TSFFrame::TSFFrame(const int &_idx, const Eigen::Matrix4f &_pose, const v4r::DataMatrix2D<Surfel> &_sf_cloud,  bool _have_track)
 : idx(_idx), pose(_pose), delta_cloud_rgb_pose(Eigen::Matrix4f::Identity()), sf_cloud(_sf_cloud), fw_link(-1), bw_link(-1), have_track(_have_track)
{
}

TSFFrame::~TSFFrame()
{
}




/***************************************************************************************/


}












