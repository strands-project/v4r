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

#include <v4r/keypoints/Part.h>

#include <stdexcept>



namespace v4r 
{


using namespace std;


double Part::pi_2 = M_PI/2.;



/************************************************************************************
 * Constructor/Destructor
 */
Part::Part()
 : type(STATIC), idx(-1), is_hyp(false),
   pose(Eigen::Matrix4f::Identity())
{ 
}

Part::Part(Type _type)
 : type(_type), idx(-1), is_hyp(false),
   pose(Eigen::Matrix4f::Identity())
{ 
}


Part::~Part()
{
}

/***************************************************************************************/


/**
 * updatePose
 */
void Part::updatePose(const Eigen::Matrix4f &_pose)
{
  pose = _pose;
}



}












