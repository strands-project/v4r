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

#include "fitting_surface_depth_im_lm.h"
#include <stdexcept>
#include <iostream>

using namespace pcl;
using namespace on_nurbs;



FittingSurfaceDepthLM::FittingSurfaceDepthLM(Parameter params)
{
  m_params = params;
  m_ddelta = 1.0/m_params.delta;
  m_huber_norm = 1.0 / ( m_params.delta*m_params.delta*(sqrt(1.0+(m_ddelta*m_ddelta))-1.0) );
}

void FittingSurfaceDepthLM::Huber(const double& x, double& h, double& hd, const double& delta)
{
  double dnorm = 1.0 / (delta*(1.0-0.5*delta));
  double a = std::abs<double>(x);
  if(a<=delta)
  {
    h = 0.5 * a * a * dnorm;
    hd = x * dnorm;
  }
  else
  {
    h = delta * (a - 0.5*delta) * dnorm;
    if(x<0.0)
      hd = -delta * dnorm;
    else
      hd = delta * dnorm;
  }
}

void FittingSurfaceDepthLM::Huber(const double& x, double &h, double& hd,
                                  const double& delta, const double& ddelta, const double& norm)
{
  double a = sqrt(1.0+(x*ddelta)*(x*ddelta));
  h = norm * delta*delta*(a-1.0);
  hd = norm * x / a;
}

void FittingSurfaceDepthLM::Huber(const Eigen::VectorXd& r,
                                  Eigen::VectorXd& h, Eigen::VectorXd& hd,
                                  double delta, double ddelta, double norm)
{
  for(unsigned j=0; j<r.rows(); j++)
    Huber(r(j,0), h(j,0), hd(j,0), delta, ddelta, norm);
}

void FittingSurfaceDepthLM::Huber(const Eigen::VectorXd& r,
                                  Eigen::VectorXd& h, SparseMatrix& Hd,
                                  double delta, double ddelta, double norm)
{
  typedef Eigen::Triplet<double> Tri;
  std::vector<Tri> tripletList;
  tripletList.resize( r.rows() );

  for(unsigned j=0; j<r.rows(); j++)
  {
    double hd;
    Huber(r(j,0), h(j,0), hd, delta, ddelta, norm);
    tripletList[j] = Tri(j,j,hd);
  }

  Hd.setFromTriplets(tripletList.begin(), tripletList.end());
}

void FittingSurfaceDepthLM::computeResidualAndJacobian(const Eigen::VectorXd& q, Eigen::VectorXd& rh,
                                                       Eigen::VectorXd& rs, SparseMatrix& J)
{
  rs = q-m_K*m_b; // residual with sign

  rh.resize(q.rows(),1); // huber norm of rs
  SparseMatrix Hd(q.rows(),q.rows());
  Huber( rs, rh, Hd, m_params.delta, m_ddelta, m_huber_norm );

  J = Hd*m_K;
}

void FittingSurfaceDepthLM::computeResidualAndJacobian(const Eigen::VectorXd& q, Eigen::VectorXd& rh,
                                                       Eigen::VectorXd& rs, SparseMatrix& J,
                                                       const std::vector<int>& indices)
{
  Eigen::VectorXd s = m_K*m_b;
  rs.resize(indices.size(),1); // residual with sign
  rh.resize(indices.size(),1); // huber norm of rs


  typedef Eigen::Triplet<double> Tri;
  std::vector<Tri> tripletList;
  tripletList.resize( indices.size() );

  for(size_t i=0; i<indices.size(); i++)
  {
    rs(i) = q(indices[i]) - s(i);
    double hd;
    Huber( rs(i), rh(i), hd, m_params.delta, m_ddelta, m_huber_norm);
    tripletList[i] = Tri(i,i,hd);
  }

  SparseMatrix Hd(indices.size(),m_K.rows());
  Hd.setFromTriplets(tripletList.begin(), tripletList.end());
  J = Hd*m_K;
}

void FittingSurfaceDepthLM::computeWeightedResidualAndJacobian(const Eigen::VectorXd& q, Eigen::VectorXd& rh,
                                                               Eigen::VectorXd& rs, SparseMatrix& J,
                                                               double sigma)
{
  // compute residual
  rs = q-m_K*m_b; // residual with sign

  double variance(sigma*sigma);

  // reweighting basis matrix K
  double d_min(DBL_MAX), d_max(DBL_MIN);
  SparseMatrix K = m_K;
  for (int k=0; k<K.outerSize(); ++k)
    for (SparseMatrix::InnerIterator it(K,k); it; ++it)
    {
      const double& d = rs(it.row());
      double w = exp(-d*d/variance);
      it.valueRef() = it.value() * w;
      if(w<d_min)
        d_min=w;
      if(w>d_max)
        d_max=w;
      if(w==0.0)
        printf("rs: %e  w: %e\n", d, w);
    }

  printf("[FittingSurfaceDepthLM::computeWeightedResidualAndJacobian] w: %e .. %e\n", d_min, d_max);

  // reweighting residual rs
  for(int i=0; i<rs.size(); i++)
  {
    const double& d = rs(i);
    rs(i) *= exp(-d*d/variance);
  }


  rh.resize(q.rows(),1); // huber norm of rs
  SparseMatrix Hd(q.rows(),q.rows());
  Huber( rs, rh, Hd, m_params.delta, m_ddelta, m_huber_norm );

  // compute jacobian
  J = Hd*K;
}

void FittingSurfaceDepthLM::computeWeightedResidualAndJacobian(const Eigen::VectorXd& q, Eigen::VectorXd& rh,
                                                               Eigen::VectorXd& rs, SparseMatrix& J,
                                                               double sigma,
                                                               const std::vector<int>& indices)
{
  // compute residual
  Eigen::VectorXd s = m_K*m_b;
  rs.resize(indices.size(),1); // residual with sign

  for(size_t i=0; i<indices.size(); i++)
    rs(i) = q(indices[i]) - s(i);

  double variance(sigma*sigma);

  // reweighting basis matrix K
  SparseMatrix K = m_K;
  for (int k=0; k<K.outerSize(); ++k)
    for (SparseMatrix::InnerIterator it(K,k); it; ++it)
    {
      const double& d = rs(it.row());
      it.valueRef() = it.value() * exp(-d*d/variance);
    }

  // reweighting residual rs
  for(size_t i=0; i<indices.size(); i++)
  {
    const double& d = rs(i);
    rs(i) *= exp(-d*d/variance);
  }

  rh.resize(indices.size(),1); // huber norm of rs
  SparseMatrix Hd(indices.size(),m_K.rows());
  Huber( rs, rh, Hd, m_params.delta, m_ddelta, m_huber_norm );

  // compute jacobian
  J = Hd*K;
}

void FittingSurfaceDepthLM::computeResidualAndJacobian(const Eigen::VectorXd& q,
                                                       Eigen::VectorXd& Jtr,
                                                       SparseMatrix& JtJ_lI)
{
  //  // compute residual
  //  rs = q-m_K*m_b; // residual with sign
  //  rh = Eigen::VectorXd(q.rows(),1); // huber norm of rs
  //  Huber( rs, rh, m_params.delta, m_ddelta, m_huber_norm );

  //  // compute jacobian
  //  SparseMatrix H(q.rows(),q.rows());
  //  HuberDerivative(rs, H, m_ddelta, m_huber_norm);
  //  J = H*m_K;

  // ----------------------------------------------
  m_rs = q-m_K*m_b;


}

void FittingSurfaceDepthLM::initSolver()
{
  if(m_nurbs.CVCount() <= 0)
    throw std::runtime_error("[FittingSurfaceDepthLM::initSolver] Error, surface not initialized (initSurface).");

  m_I = SparseMatrix(m_nurbs.CVCount(),m_nurbs.CVCount());
  m_K = SparseMatrix( m_roi.width*m_roi.height, m_nurbs.CVCount() );

  typedef Eigen::Triplet<double> Tri;
  std::vector<Tri> tripletList;
  tripletList.resize( m_roi.width * m_roi.height * m_nurbs.Order(0) * m_nurbs.Order(1) );

  printf("[FittingSurfaceDepthLM::initSolver] entries: %d  rows: %d  cols: %d\n",
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
          ti++;
        } // j
      } // i

    } // u
  } // v

  delete [] N1;
  delete [] N0;

  m_K.setFromTriplets(tripletList.begin(), tripletList.end());

  m_use_indices = false;
}

void FittingSurfaceDepthLM::initSolver(const std::vector<int>& indices)
{
  if(m_nurbs.CVCount() <= 0)
    throw std::runtime_error("[FittingSurfaceDepth::initSolver] Error, surface not initialized (initSurface).");

  m_I = SparseMatrix(m_nurbs.CVCount(),m_nurbs.CVCount());
  m_K = SparseMatrix( indices.size(), m_nurbs.CVCount() );

  typedef Eigen::Triplet<double> Tri;
  std::vector<Tri> tripletList;
  tripletList.resize( indices.size() * m_nurbs.Order(0) * m_nurbs.Order(1) );

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
    u = indices[row] % m_roi.width;
    v = indices[row] / m_roi.width;

    E = ON_NurbsSpanIndex (m_nurbs.m_order[0], m_nurbs.m_cv_count[0], m_nurbs.m_knot[0], double(u), 0, 0);
    F = ON_NurbsSpanIndex (m_nurbs.m_order[1], m_nurbs.m_cv_count[1], m_nurbs.m_knot[1], double(v), 0, 0);

    ON_EvaluateNurbsBasis (m_nurbs.Order (0), m_nurbs.m_knot[0] + E, double(u), N0);
    ON_EvaluateNurbsBasis (m_nurbs.Order (1), m_nurbs.m_knot[1] + F, double(v), N1);

    for (int i = 0; i < m_nurbs.Order (0); i++)
    {
      for (int j = 0; j < m_nurbs.Order (1); j++)
      {
        tripletList[ti] = Tri( row, lrc2gl (E, F, i, j), N0[i] * N1[j] );
        ti++;
      } // j
    } // i


  } // row

  delete [] N1;
  delete [] N0;

  m_K.setFromTriplets(tripletList.begin(), tripletList.end());

  m_use_indices = true;
}

#include "boost/date_time/posix_time/posix_time_types.hpp"

void FittingSurfaceDepthLM::solve(Eigen::VectorXd& z, int iterations)
{
  if(m_use_indices)
    throw std::runtime_error("[FittingSurfaceDepthLM::solve] Error, solver initialized with indices (use solve(Eigen::VectorXd&, const std::vector<int>&) instead.\n");


  Eigen::VectorXd rs, rh, f, step;
  SparseMatrix J, A;

  //  for(unsigned i=0;
  //      i<m_params.max_iterations &&
  //      step.norm()>m_params.threshold;
  //      i++)
  //  {

  {
    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();

    if(iterations>0 && m_params.delta>0.0)
      computeWeightedResidualAndJacobian(z, rh, rs, J, m_params.delta);
    else
      computeResidualAndJacobian(z, rh, rs, J);

    // Levenberg-Marquardt
    A = J.transpose()*J + m_params.lambda * m_I;
    f = J.transpose()*rh;

    for (int k=0; k<A.outerSize(); ++k)
      for (SparseMatrix::InnerIterator it(A,k); it; ++it)
      {
        if(it.value()<0.0)
          throw std::runtime_error("[FittingSurfaceDepthLM::solve] Error, value < 0.");

      }

    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration dt = t2 - t1;
    std::cout << "[FittingSurfaceDepthLM::solve] ASSEMBLY took: ";
    std::cout << dt.total_milliseconds() << " ms" << std::endl;
  }

  // solve for step width
  //  {
  //    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
  //    m_solverSimplicialLLT.compute(A);
  //    step = m_solverSimplicialLLT.solve(f);
  //    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
  //    boost::posix_time::time_duration dt = t2 - t1;
  //    std::cout << "[FittingSurfaceDepthLM::solve] SimplicialLLT took: ";
  //    std::cout << dt.total_milliseconds() << " ms" << std::endl;
  //  }

  //  {
  //    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
  //    m_solverSimplicialLDLT.compute(A);
  //    step = m_solverSimplicialLDLT.solve(f);
  //    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
  //    boost::posix_time::time_duration dt = t2 - t1;
  //    std::cout << "[FittingSurfaceDepthLM::solve] SimplicialLDLT took: ";
  //    std::cout << dt.total_milliseconds() << " ms" << std::endl;
  //  }

  //  {
  //    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
  //    m_solverConjugate.compute(A);
  //    step = m_solverConjugate.solve(f);
  //    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
  //    boost::posix_time::time_duration dt = t2 - t1;
  //    std::cout << "[FittingSurfaceDepthLM::solve] ConjugateGradient took: ";
  //    std::cout << dt.total_milliseconds() << " ms" << std::endl;
  //  }

//    {
//      boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
//      m_solver = new SPQR();
//      m_solver->compute(A);
//      step = m_solver->solve(f);
//      delete m_solver;
//      m_solver = NULL;
//      boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
//      boost::posix_time::time_duration dt = t2 - t1;
//      std::cout << "[FittingSurfaceDepthLM::solve] SPQR took: ";
//      std::cout << dt.total_milliseconds() << " ms" << std::endl;
//    }

  //  {
  //    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
  //    m_solverSparseLU.compute(A);
  //    step = m_solverSparseLU.solve(f);
  //    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
  //    boost::posix_time::time_duration dt = t2 - t1;
  //    std::cout << "[FittingSurfaceDepthLM::solve] SparseLU took: ";
  //    std::cout << dt.total_milliseconds() << " ms" << std::endl;
  //  }

//    {
//      boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
//      m_solverUmfPack.compute(A);
//      step = m_solverUmfPack.solve(f);
//      boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
//      boost::posix_time::time_duration dt = t2 - t1;
//      std::cout << "[FittingSurfaceDepthLM::solve] UmfPackLU took: ";
//      std::cout << dt.total_milliseconds() << " ms" << std::endl;
//    }

  {
    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
    m_solverCholmod.compute(A);
    step = m_solverCholmod.solve(f);
    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration dt = t2 - t1;
    std::cout << "[FittingSurfaceDepthLM::solve] Cholmod took: ";
    std::cout << dt.total_milliseconds() << " ms" << std::endl;
  }

  m_b = m_b + step;
  //  }

  updateSurf();
}

void FittingSurfaceDepthLM::solve(Eigen::VectorXd &z, const std::vector<int>& indices, int iterations)
{
  if(!m_use_indices)
    throw std::runtime_error("[FittingSurfaceDepthLM::solve] Error, solver initialized without indices (use solve(Eigen::VectorXd&) instead.\n");

  Eigen::VectorXd rs, rh, f, step;
  SparseMatrix J, A;

  {
    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();

    if(iterations>0 && m_params.delta>0.0)
      computeWeightedResidualAndJacobian(z, rh, rs, J, m_params.delta, indices);
    else
      computeResidualAndJacobian(z, rh, rs, J, indices);

    // Levenberg-Marquardt
    A = J.transpose()*J + m_params.lambda * m_I;
    f = J.transpose()*rh;

    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration dt = t2 - t1;
    std::cout << "[FittingSurfaceDepthLM::solve] ASSEMBLY took: ";
    std::cout << dt.total_milliseconds() << " ms" << std::endl;
  }

  // solve for step width
  {
    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
    m_solverCholmod.compute(A);
    step = m_solverCholmod.solve(f);
    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration dt = t2 - t1;
    std::cout << "[FittingSurfaceDepthLM::solve] Cholmod took: ";
    std::cout << dt.total_milliseconds() << " ms" << std::endl;
  }

  m_b = m_b + step;

  updateSurf();

}
