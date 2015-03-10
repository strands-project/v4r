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

#ifndef NURBS_FITTING_SURFACE_DEPTH_IM_H
#define NURBS_FITTING_SURFACE_DEPTH_IM_H

#include "nurbs_tools.h"
#include "nurbs_data.h"
#include "nurbs_solve.h"

#include <Eigen/SparseQR>
#include <Eigen/SPQRSupport>

namespace pcl
{
namespace on_nurbs
{

/** \brief      */
class FittingSurfaceDepthIM
{
public:

  struct ROI
  {
    int x, y, width, height;
    ROI() : x(0), y(0), width(0), height(0) {}
    ROI(int _x, int _y, int _w, int _h) : x(_x), y(_y), width(_w), height(_h) {}
    ROI(const std::vector<int>& indices, int cloud_width);
  };

protected:
  Eigen::VectorXd m_b; // control points
  ON_NurbsSurface m_nurbs;
  ROI m_roi;

  typedef Eigen::SparseMatrix<double> SparseMatrix;
  SparseMatrix m_K;
  Eigen::VectorXd m_rs;

  bool m_quiet;

  typedef Eigen::SPQR<SparseMatrix> SPQR;
  SPQR* m_solver;

  bool m_use_indices;

  void updateSurf();

public:
  FittingSurfaceDepthIM() : m_quiet(true), m_solver(NULL) {}
  FittingSurfaceDepthIM(int order,
                      int cps_width, int cps_height,
                      ROI img_roi,
                      const Eigen::VectorXd& z);
  FittingSurfaceDepthIM(int order,
                      int cps_width, int cps_height,
                      ROI img_roi,
                      const Eigen::VectorXd& z,
                      const std::vector<int>& indices, int data_width);
  ~FittingSurfaceDepthIM();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline ON_NurbsSurface& getSurface ()
  {
    return m_nurbs;
  }

  void initSurface(int order, int cps_width, int cps_height, ROI img_roi);

  virtual void initSolver();
  virtual void initSolver(const std::vector<int>& indices, int data_width);

  virtual void solve(const Eigen::VectorXd& z);
  virtual void solve(const Eigen::VectorXd& z, const std::vector<int>& indices);

  virtual void solveReweight(const Eigen::VectorXd &z);
  virtual void solveReweight(const Eigen::VectorXd &z, double variance);

  virtual void solveLP(const Eigen::VectorXd& z);
  virtual void solveLP(const Eigen::VectorXd& z, const std::vector<int>& indices);

  virtual void solveGurobi(const Eigen::VectorXd &z);

  Eigen::VectorXd GetError(const Eigen::VectorXd& z);
  Eigen::VectorXd GetError(const Eigen::VectorXd& z, const std::vector<int>& indices);



  // index routines
  int grc2gl (int I, int J)
  {
    return m_nurbs.CVCount (1) * I + J;
  } // global row/col index to global lexicographic index

  int lrc2gl (int E, int F, int i, int j)
  {
    return grc2gl (E + i, F + j);
  } // local row/col index to global lexicographic index

  int gl2gr (int A)
  {
    return (static_cast<int> (A / m_nurbs.CVCount (1)));
  } // global lexicographic in global row index

  int gl2gc (int A)
  {
    return (static_cast<int> (A % m_nurbs.CVCount (1)));
  } // global lexicographic in global col index

};
}
}

#endif
