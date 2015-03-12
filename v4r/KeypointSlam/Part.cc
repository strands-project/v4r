/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "Part.hh"

#include <stdexcept>



namespace kp 
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












