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
 *
 *
 *
 */

#ifndef FITTING_SURFACE_DEPTH_IM_LM_H
#define FITTING_SURFACE_DEPTH_IM_LM_H

#include "fitting_surface_depth_im.h"

#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/UmfPackSupport>

namespace pcl
{
namespace on_nurbs
{

class FittingSurfaceDepthLM : public FittingSurfaceDepthIM
{
public:
  struct Parameter
  {
    double delta;
    double lambda;
    double threshold;
    unsigned max_iterations;
  };

 protected:
  Parameter m_params;
  double m_ddelta;
  double m_huber_norm;
  SparseMatrix m_I;

  Eigen::SimplicialLLT<SparseMatrix> m_solverSimplicialLLT;
  Eigen::SimplicialLDLT<SparseMatrix> m_solverSimplicialLDLT;
  Eigen::ConjugateGradient<SparseMatrix> m_solverConjugate;
  Eigen::SparseLU<SparseMatrix> m_solverSparseLU;
  Eigen::UmfPackLU<SparseMatrix> m_solverUmfPack;
  Eigen::CholmodSupernodalLLT<SparseMatrix> m_solverCholmod;

  // FittingSurfaceDepth::K is used to store dr/dB (jacobian = d|r|/dr * dr/db)
  // rs is the signed L1 distance of the surface from the depth values; b are the control points
  // rh is the huber norm of rs
  // J and K is an MxN matrix where M is the number of residuals and N the number of control points
  // d|r|/dr is computed w.r.t. normalized pseudo Huber norm (NPHN) and is MxM, i.e. diag(d|r_j|/dr_j)
  // q is the vector of depths, then the residual is |K*b-q| w.r.t. NPHN
  void computeResidualAndJacobian(const Eigen::VectorXd &q, Eigen::VectorXd& rh,
                                  Eigen::VectorXd& rs, SparseMatrix& J);
  void computeResidualAndJacobian(const Eigen::VectorXd& q, Eigen::VectorXd& rh,
                                  Eigen::VectorXd& rs, SparseMatrix& J,
                                  const std::vector<int>& indices);


  void computeWeightedResidualAndJacobian(const Eigen::VectorXd& q, Eigen::VectorXd& rh,
                                          Eigen::VectorXd& rs, SparseMatrix& J,
                                          double sigma);
  void computeWeightedResidualAndJacobian(const Eigen::VectorXd& q, Eigen::VectorXd& rh,
                                          Eigen::VectorXd& rs, SparseMatrix& J,
                                          double sigma,
                                          const std::vector<int>& indices);

  void computeResidualAndJacobian(const Eigen::VectorXd& q, Eigen::VectorXd& Jtr, SparseMatrix& JtJ_lI);

public:
  static void Huber(const double& x, double& h, double& hd, const double& delta);
  static void Huber(const double& x, double &h, double& hd, const double& delta, const double& ddelta, const double& norm);
  static void Huber(const Eigen::VectorXd& r, Eigen::VectorXd &h, Eigen::VectorXd &hd, double delta, double ddelta, double norm);
  static void Huber(const Eigen::VectorXd& r, Eigen::VectorXd &h, SparseMatrix& hd, double delta, double ddelta, double norm);

public:
  FittingSurfaceDepthLM(Parameter params);

  virtual void initSolver();
  virtual void initSolver(const std::vector<int>& indices);

  virtual void solve(Eigen::VectorXd& z, int iterations);
  virtual void solve(Eigen::VectorXd &z, const std::vector<int>& indices, int iterations);


};

} // namespace on_nurbs
} // namespace pcl

#endif // FITTING_SURFACE_DEPTH_LM_H
