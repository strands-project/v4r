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

#include "fitting_surface_depth_im.h"
#include <stdexcept>

//#include <glpk.h>
//#include <gurobi/gurobi_c++.h>

using namespace pcl;
using namespace on_nurbs;

FittingSurfaceDepthIM::ROI::ROI(const std::vector<int>& indices, int cloud_width)
{
  int u_min(INT_MAX), u_max(INT_MIN);
  int v_min(INT_MAX), v_max(INT_MIN);

  for(size_t i=0; i<indices.size(); i++)
  {
    const int& idx = indices[i];
    int u = (idx % cloud_width);
    int v = (idx / cloud_width);
    if(u<u_min)
      u_min=u;
    if(u>u_max)
      u_max=u;
    if(v<v_min)
      v_min=v;
    if(v>v_max)
      v_max=v;
  }

  x = u_min;
  y = v_min;
  width = u_max-u_min;
  height = v_max-v_min;
}

FittingSurfaceDepthIM::FittingSurfaceDepthIM(int order,
                                         int cps_width, int cps_height,
                                         ROI img_roi,
                                         const Eigen::VectorXd& z)
  : m_quiet(true), m_solver(NULL)
{
  initSurface(order, cps_width, cps_height, img_roi);
  initSolver();
  solve(z);
}

FittingSurfaceDepthIM::FittingSurfaceDepthIM(int order,
                                         int cps_width, int cps_height,
                                         ROI img_roi,
                                         const Eigen::VectorXd& z,
                                         const std::vector<int>& indices, int data_width)
  : m_quiet(true), m_solver(NULL)
{
  initSurface(order, cps_width, cps_height, img_roi);
  initSolver(indices, data_width);
  solve(z,indices);
}

FittingSurfaceDepthIM::~FittingSurfaceDepthIM()
{
  if(m_solver!=NULL)
    delete m_solver;
}


void FittingSurfaceDepthIM::initSurface(int order, int cps_width, int cps_height, ROI img_roi)
{
  if(cps_width<order)
    cps_width=order;
  if(cps_height<order)
    cps_height=order;

  m_roi = img_roi;
  m_nurbs = ON_NurbsSurface (1, false, order, order, cps_width, cps_height);

  double ddx = double(m_roi.width+1)  / (m_nurbs.KnotCount(0) - 2*(order-2) - 1);
  double ddy = double(m_roi.height+1) / (m_nurbs.KnotCount(1) - 2*(order-2) - 1);

  m_nurbs.MakeClampedUniformKnotVector (0, ddx);
  m_nurbs.MakeClampedUniformKnotVector (1, ddy);

  for (int i = 0; i < m_nurbs.KnotCount(0); i++)
  {
    double k = m_nurbs.Knot (0, i);
    m_nurbs.SetKnot (0, i, k + m_roi.x-1);
  }

  for (int i = 0; i < m_nurbs.KnotCount(1); i++)
  {
    double k = m_nurbs.Knot (1, i);
    m_nurbs.SetKnot (1, i, k + m_roi.y-1);
  }

  m_b = Eigen::VectorXd(m_nurbs.CVCount(),1);
  for (int i = 0; i < m_nurbs.CVCount(0); i++)
  {
    for (int j = 0; j < m_nurbs.CVCount(1); j++)
    {
      m_nurbs.SetCV (i, j, ON_3dPoint(0,0,0));
      m_b(grc2gl(i,j),0) = 0.0;
    }
  }

//    ON_TextLog out;
//    m_nurbs.Dump (out);
}

void FittingSurfaceDepthIM::initSolver()
{
  if(m_nurbs.CVCount() <= 0)
    throw std::runtime_error("[FittingSurfaceDepth::initSolver] Error, surface not initialized (initSurface).");

  m_K = SparseMatrix( m_roi.width*m_roi.height, m_nurbs.CVCount() );

  typedef Eigen::Triplet<double> Tri;
  std::vector<Tri> tripletList;
  tripletList.resize( m_roi.width * m_roi.height * m_nurbs.Order(0) * m_nurbs.Order(1) );

  if(!m_quiet)
    printf("[FittingSurfaceDepth::initSolver] entries: %d  rows: %d  cols: %d\n",
           m_roi.width * m_roi.height * m_nurbs.Order(0) * m_nurbs.Order(1),
           m_roi.width * m_roi.height,
           m_nurbs.CVCount());

  double *N0 = new double[m_nurbs.Order (0) * m_nurbs.Order (0)];
  double *N1 = new double[m_nurbs.Order (1) * m_nurbs.Order (1)];
  int E,F,row;
  int ti(0);

  for(int v=0; v<m_roi.height; v++)
  {
    for(int u=0; u<m_roi.width; u++)
    {
      row = v * m_roi.width + u;

      E = ON_NurbsSpanIndex (m_nurbs.m_order[0], m_nurbs.m_cv_count[0], m_nurbs.m_knot[0], double(u), 0, 0);
      F = ON_NurbsSpanIndex (m_nurbs.m_order[1], m_nurbs.m_cv_count[1], m_nurbs.m_knot[1], double(v), 0, 0);

      ON_EvaluateNurbsBasis (m_nurbs.Order (0), m_nurbs.m_knot[0] + E, double(u), N0);
      ON_EvaluateNurbsBasis (m_nurbs.Order (1), m_nurbs.m_knot[1] + F, double(v), N1);

      for (int i = 0; i < m_nurbs.Order (0); i++)
      {
        for (int j = 0; j < m_nurbs.Order (1); j++)
        {
          tripletList[ti] = Tri( row, lrc2gl (E, F, i, j), N0[i] * N1[j] );
          //          m_nsolver.K(ti,lrc2gl (E, F, i, j),N0[i]*N1[j]);
          ti++;
        } // j
      } // i

    } // u
  } // v

  delete [] N1;
  delete [] N0;

  m_K.setFromTriplets(tripletList.begin(), tripletList.end());

  if(m_solver==NULL)
    m_solver = new SPQR();
  m_solver->compute(m_K);
  if(m_solver->info()!=Eigen::Success)
    throw std::runtime_error("[FittingSurfaceDepth::initSolver] decomposition failed.");

  if(!m_quiet)
    printf("[FittingSurfaceDepth::initSolver] decomposition done\n");

  m_use_indices = false;
}

void FittingSurfaceDepthIM::initSolver(const std::vector<int>& indices, int data_width)
{
  if(m_nurbs.CVCount() <= 0)
    throw std::runtime_error("[FittingSurfaceDepth::initSolver] Error, surface not initialized (initSurface).");

  //  m_nsolver.assign(m_roi.width*m_roi.height, m_nurbs.CVCount(), 1);

  m_K = SparseMatrix( indices.size(), m_nurbs.CVCount() );

  typedef Eigen::Triplet<double> Tri;
  std::vector<Tri> tripletList;
  tripletList.resize( indices.size() * m_nurbs.Order(0) * m_nurbs.Order(1) );

  if(!m_quiet)
    printf("[FittingSurfaceDepth::initSolver] entries: %lu  rows: %lu  cols: %d\n",
           indices.size() * m_nurbs.Order(0) * m_nurbs.Order(1),
           indices.size(),
           m_nurbs.CVCount());

  double *N0 = new double[m_nurbs.Order (0) * m_nurbs.Order (0)];
  double *N1 = new double[m_nurbs.Order (1) * m_nurbs.Order (1)];
  int E,F, u,v;
  int ti(0);

  for(size_t row=0; row<indices.size(); row++)
  {
    u = indices[row] % data_width;
    v = indices[row] / data_width;

    E = ON_NurbsSpanIndex (m_nurbs.m_order[0], m_nurbs.m_cv_count[0], m_nurbs.m_knot[0], double(u), 0, 0);
    F = ON_NurbsSpanIndex (m_nurbs.m_order[1], m_nurbs.m_cv_count[1], m_nurbs.m_knot[1], double(v), 0, 0);

    ON_EvaluateNurbsBasis (m_nurbs.Order (0), m_nurbs.m_knot[0] + E, double(u), N0);
    ON_EvaluateNurbsBasis (m_nurbs.Order (1), m_nurbs.m_knot[1] + F, double(v), N1);

    for (int i = 0; i < m_nurbs.Order (0); i++)
    {
      for (int j = 0; j < m_nurbs.Order (1); j++)
      {
        tripletList[ti] = Tri( row, lrc2gl (E, F, i, j), N0[i] * N1[j] );
        //          m_nsolver.K(ti,lrc2gl (E, F, i, j),N0[i]*N1[j]);
        ti++;
      } // j
    } // i


  } // row

  delete [] N1;
  delete [] N0;

  m_K.setFromTriplets(tripletList.begin(), tripletList.end());

  if(m_solver==NULL)
    m_solver = new SPQR();
  m_solver->compute(m_K);
  if(m_solver->info()!=Eigen::Success)
    throw std::runtime_error("[FittingSurfaceDepth::initSolver] decomposition failed.");
  if(!m_quiet)
    printf("[FittingSurfaceDepth::initSolver] decomposition done\n");
  m_use_indices = true;
}

void FittingSurfaceDepthIM::solve(const Eigen::VectorXd &z)
{
  if(m_use_indices)
    throw std::runtime_error("[FittingSurfaceDepth::solve] Error, solver initialized with indices (use solve(Eigen::VectorXd&, const std::vector<int>&) instead.\n");

  m_b = m_solver->solve(z);

  updateSurf();
}

void FittingSurfaceDepthIM::solveReweight(const Eigen::VectorXd &z)
{
  SparseMatrix A = m_K;
  Eigen::VectorXd zn(z.rows(),1);
  Eigen::VectorXd r = GetError(z);

  // compute variance
  double variance(0.0);
  for(Eigen::VectorXd::Index i=0; i<r.rows(); i++)
  {
    const double& d = r(i);
    variance += d*d;
  }
  variance /= r.rows();

  for (int k=0; k<A.outerSize(); ++k)
  for (SparseMatrix::InnerIterator it(A,k); it; ++it)
  {
    const double& d = r(it.row());
    double w = exp(-d*d/(variance)); // normal distribution
    it.valueRef() = it.value() * w;
    zn(it.row()) = z(it.row()) *  w;
  }

  m_solver->compute(A);
  m_b = m_solver->solve(zn);

//  Eigen::VectorXd bn = m_solver.solve(zn);
//  double delta = (bn-m_b).norm();
//  m_b = bn;
//  if(!m_quiet)
//    printf("[FittingSurfaceDepth::solveReweight] delta: %f\n", delta);

  updateSurf();
}

void FittingSurfaceDepthIM::solveReweight(const Eigen::VectorXd &z, double variance)
{
  SparseMatrix A = m_K;
  Eigen::VectorXd zn(z.rows(),1);
  Eigen::VectorXd r = GetError(z);

  for (int k=0; k<A.outerSize(); ++k)
  for (SparseMatrix::InnerIterator it(A,k); it; ++it)
  {
    const double& d = r(it.row());
    double w = exp(-d*d/(variance)); // normal distribution
    it.valueRef() = it.value() * w;
    zn(it.row()) = z(it.row()) *  w;
  }

  m_solver->compute(A);
  m_b = m_solver->solve(zn);

  updateSurf();
}

void FittingSurfaceDepthIM::solve(const Eigen::VectorXd &z, const std::vector<int>& indices)
{
  if(!m_use_indices)
    throw std::runtime_error("[FittingSurfaceDepth::solve] Error, solver initialized without indices (use solve(Eigen::VectorXd&) instead.\n");

  Eigen::VectorXd zn(m_solver->rows(),1);

  for(size_t i=0; i<indices.size(); i++)
    zn(i) = z(indices[i]);

  m_b = m_solver->solve(zn);

  updateSurf();
}

void FittingSurfaceDepthIM::solveLP(const Eigen::VectorXd &z)
{
  if(m_use_indices)
    throw std::runtime_error("[FittingSurfaceDepth::solve] Error, solver initialized with indices (use solve(Eigen::VectorXd&, const std::vector<int>&) instead.\n");

//  glp_prob *lp;
//  size_t non_zeros = 2*(m_K.rows() + m_K.nonZeros());
//  std::vector<int> row_indices(1+non_zeros);
//  std::vector<int> col_indices(1+non_zeros);
//  std::vector<double> values(1+non_zeros);

//  size_t m = static_cast<size_t>(m_K.rows());
//  size_t n = static_cast<size_t>(m_K.cols());

//  if(!m_quiet)
//    printf("[FittingSurfaceDepth::solveLP] m: %lu  n: %lu  non_zeros: %lu\n", m, n, non_zeros);

//  // solve L1 as linear programm
//  // minimize [x b]^T * [e 0]
//  // subjet to    |I -K| |x| >= |-z|
//  //              |I  K| |b|    | z|

//  // assign identity matrices I element of R^{mxm}
//  for(size_t i=0; i<m; i++)
//  {
//    row_indices[1+i] = i+1;
//    col_indices[1+i] = i+1;
//    values[1+i] = 1.0;

//    col_indices[1+m+i] = i+1;
//    row_indices[1+m+i] = m+i+1;
//    values[1+m+i] = 1.0;
//  }

//  // assign basis matrix -K and K
//  size_t idx = 2*m;
//  size_t stride = m_K.nonZeros();
//  for (int k=0; k<m_K.outerSize(); ++k)
//  for (SparseMatrix::InnerIterator it(m_K,k); it; ++it)
//  {
//    // -K
//    row_indices[1+idx] = it.row()+1;
//    col_indices[1+idx] = m + it.col()+1;
//    values[1+idx] = -it.value();

//    // K
//    row_indices[1+idx+stride] = m + it.row()+1;
//    col_indices[1+idx+stride] = m + it.col()+1;
//    values[1+idx+stride] = it.value();

//    idx++;
//  }

//  // assemble linear program
//  lp = glp_create_prob();
//  glp_set_obj_dir(lp, GLP_MIN);
//  glp_add_rows(lp, m+m);
//  glp_add_cols(lp, m+n);

//  // set z vector as lower bound (as defined in 'subject to')
//  for(size_t i=0; i<m; i++)
//  {
//    glp_set_row_bnds(lp, i+1, GLP_LO,-z(i),0.0);
//    glp_set_row_bnds(lp,m+i+1, GLP_LO,z(i),0.0);
//  }

//  // constraints for solution vector [x b]
//  for(size_t i=0; i<m; i++)
//  {
//    glp_set_col_bnds(lp, i+1,  GLP_UP, 0.0, 0.0);
//    glp_set_col_bnds(lp, i+1,  GLP_LO, 0.0, 0.0);
//  }
//  for(size_t i=0; i<n; i++)
//  {
//    glp_set_col_bnds(lp, m+i+1, GLP_LO, 0.0, 0.0);
//  }

//  // set coefficence for objective function, i.e. [1 0] (1 for x- and 0 for b-vector)
//  for(size_t i=0; i<m; i++)
//    glp_set_obj_coef(lp,i+1,1.0);
//  for(size_t i=0; i<n; i++)
//    glp_set_obj_coef(lp,m+i+1,0.0);

//  // set coefficence for constraints
//  glp_load_matrix(lp,non_zeros,&row_indices[0],&col_indices[0],&values[0]);

//  // solve
//  glp_simplex(lp, NULL);

//  // get control points
//  for(size_t i=0; i<n; i++)
//  {
//    m_b(i) = glp_get_col_prim(lp, m+i+1);
//  }

//  glp_delete_prob(lp);

//  updateSurf();

}

void FittingSurfaceDepthIM::solveLP(const Eigen::VectorXd &z, const std::vector<int>& indices)
{
  if(!m_use_indices)
    throw std::runtime_error("[FittingSurfaceDepth::solve] Error, solver initialized without indices (use solve(Eigen::VectorXd&) instead.\n");


}

void FittingSurfaceDepthIM::solveGurobi(const Eigen::VectorXd &z)
{
//  GRBenv* env = NULL;
//  int grb_err = 0;

//  grb_err = GRBloadenv(env, "grb.log");
//  GRBModel model = new GRBModel(env);





//  GRBfreemodel(model);
//  GRBfreeenv(env);

}

void FittingSurfaceDepthIM::updateSurf()
{
  int ncp = m_nurbs.CVCount ();

  if(m_b.rows()!=ncp)
    throw std::runtime_error("[FittingSurfaceDepth::updateSurf] Error, number of control points does not match.");

  for (int i = 0; i < ncp; i++)
  {
    ON_3dPoint cp;

    cp.x = m_b(i, 0);
    cp.y = cp.x;
    cp.z = cp.x;

    m_nurbs.SetCV (gl2gr(i), gl2gc(i), cp);
  }
}

Eigen::VectorXd FittingSurfaceDepthIM::GetError(const Eigen::VectorXd& z)
{
  if(m_use_indices)
    throw std::runtime_error("[FittingSurfaceDepth::GetError] Error, solver initialized with indices (use solve(Eigen::VectorXd&, const std::vector<int>&) instead.\n");

  // compute K*b (i.e. points on surface)
  Eigen::VectorXd s(z.rows(),1);
  s.setZero();
  for (int k=0; k<m_K.outerSize(); ++k)
  for (SparseMatrix::InnerIterator it(m_K,k); it; ++it)
    s(it.row()) += it.value() * m_b(it.col());

  // return (K*b-z)
  return (s-z);
}

Eigen::VectorXd FittingSurfaceDepthIM::GetError(const Eigen::VectorXd& z, const std::vector<int>& indices)
{
  if(!m_use_indices)
    throw std::runtime_error("[FittingSurfaceDepth::GetError] Error, solver initialized without indices (use solve(Eigen::VectorXd&) instead.\n");

  // compute K*b (i.e. points on surface)
  Eigen::VectorXd e(indices.size(),1);
  e.setZero();
  for (int k=0; k<m_K.outerSize(); ++k)
  for (SparseMatrix::InnerIterator it(m_K,k); it; ++it)
    e(it.row()) += it.value() * m_b(it.col());

  // compute (K*b-z)
  for(size_t i=0; i<indices.size(); i++)
    e(i) -= z(indices[i]);

  return e;
}
