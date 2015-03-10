/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012-, Thomas Mörwald
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#include "fitting_plane_depth.h"
#include <Eigen/Eigenvalues>

using namespace pcl;
using namespace on_nurbs;

Eigen::Vector3d FittingDepthPlane::computeMean(Eigen::VectorXd& depth, Eigen::VectorXd::Index width)
{
  Eigen::Vector3d m(0,0,0);
  for(Eigen::VectorXd::Index i=0; i<depth.rows(); i++)
  {
    m(0) += double(i % width);
    m(1) += double(i / width);
    m(2) += depth(i);
  }
  return (m/depth.rows());
}

FittingDepthPlane::FittingDepthPlane(Eigen::VectorXd& depth, Eigen::VectorXd::Index width)
{
  Eigen::Vector3d mean = computeMean(depth, width);
  Eigen::MatrixXd Q(depth.rows(), 3);

  for(Eigen::VectorXd::Index i=0; i<depth.rows(); i++)
  {
    Q(i,0) = double(i % width) - mean(0);
    Q(i,1) = double(i / width) - mean(1);
    Q(i,2) = depth(i) - mean(2);
  }

  Eigen::MatrixXd C = Q.transpose() * Q;

  Eigen::EigenSolver<Eigen::MatrixXd> es(C);

  // get smallest eigenvalue
  Eigen::VectorXd::Index j = 0;
  if(es.eigenvalues()[1].real() <  es.eigenvalues()[0].real())
    j = 1;
  if(es.eigenvalues()[2].real() <  es.eigenvalues()[j].real())
    j = 2;

  // plane coefficients
  double a = es.eigenvectors()(0,j).real();
  double b = es.eigenvectors()(1,j).real();
  double c = es.eigenvectors()(2,j).real();
  double d = -a*mean(0)-b*mean(1)-c*mean(2);

  // divided by z coefficient
  m_ac = a/c;
  m_bc = b/c;
  m_dc = d/c;
}

double FittingDepthPlane::evaluate(double u, double v)
{
  return (-m_ac*u - m_bc*v - m_dc);
}


Eigen::VectorXd FittingDepthPlane::getError(Eigen::VectorXd& depth, Eigen::VectorXd::Index width)
{
  Eigen::VectorXd e(depth.rows(),1);
  for(Eigen::VectorXd::Index i=0; i<depth.rows(); i++)
    e(i) = depth(i) - evaluate( double(i%width), double(i/width) );
  return e;
}
