/*
 * lm_icp.h
 *
 *  Created on: Jun 12, 2013
 *      Author: aitor
 */

#include "faat_pcl/registration/mv_lm_icp.h"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/time.h>
#include "sparselm-1.3/splm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#undef REPEATABLE_RANDOM
#define DBL_RAND_MAX (double)(RAND_MAX)

#ifdef _MSC_VER // MSVC
#include <process.h>
#define GETPID  _getpid
#elif defined(__GNUC__) // GCC
#include <sys/types.h>
#include <unistd.h>
#define GETPID  getpid
#else
#warning Do not know the name of the function returning the process id for your OS/compiler combination
#define GETPID  0
#endif /* _MSC_VER */

#ifdef REPEATABLE_RANDOM
#define INIT_RANDOM(seed) srandom(seed)
#else
#define INIT_RANDOM(seed) srandom((int)GETPID()) // seed unused
#endif

bool MVNLICP_WEIRD_TRANS_ = false;

namespace faat_pcl
{
  namespace registration
  {
    namespace MVNLICP
    {
      inline double huber(double d, double sigma)
      {
        double k = 1.345*sigma;
        if(d <= k)
        {
          return (d*d) / 2;
        }
        else
        {
          return k * d - ((d*d) / 2);
        }
      }

      inline double huberFitzigibon(double d, double sigma)
      {
        if(d <= sigma)
        {
          return (d*d)/2.0;
        }
        else
        {
          return sigma * d - sigma * sigma / 2.0;
        }
      }

      inline double lorentzian(double d, double sigma)
      {
        return log10(1 + (d * d) / sigma);
      }

      inline Eigen::Matrix3d coolquat_to_mat(Eigen::Quaterniond & q)
      {
        Eigen::Matrix3d R;
        double x2 = q.x() * q.x();
        double y2 = q.y() * q.y();
        double z2 = q.z() * q.z();
        double r2 = q.w() * q.w();

        R(0,0) = r2 + x2 - y2 - z2;
        R(1,1) = r2 - x2 + y2 - z2;
        R(2,2) = r2 - x2 - y2 + z2;

        double xy = q.x()  * q.y();
        double yz = q.y() *  q.z();
        double zx =  q.z() * q.x();
        double rx = q.w() * q.x();
        double ry = q.w() * q.y();
        double rz = q.w() *  q.z();

        R(0,1) = 2 * (xy + rz);
        R(0,2) = 2 * (zx - ry);
        R(1,2) = 2 * (yz + rx);
        R(1,0) = 2 * (xy - rz);
        R(2,0) = 2 * (zx + ry);
        R(2,1) = 2 * (yz - rx);
        return R;
      }

      /*template <typename PointT>
      void model_func(double *p, double *x, int m, int n, void *data)
      {
        pcl::ScopeTime t("model_func");
        MVNonLinearICP<PointT> * nl_icp = (MVNonLinearICP<PointT> *)(data);
        double inliers_threshold_ = nl_icp->inlier_threshold_ * 1.345;
        //double out = inliers_threshold_ * 1.5;

        for(size_t k=0; k < n; k++)
        {
          //x[k] = huberFitzigibon(out, inliers_threshold_);
          x[k] = huberFitzigibon(nl_icp->max_correspondence_distance_, inliers_threshold_);
        }

        int k=0;
        int used=0;
        int total = 0;
        for(size_t i =0; i < nl_icp->S_.size(); i++)
        {
          int idx_match;
          float distance = 0;
          float color_distance = -1.f;

          Eigen::Matrix3d R_h, R_k;
          Eigen::Vector3d trans_h, trans_k;
          Eigen::Matrix4f T_h, T_k;
          T_h.setIdentity();
          T_k.setIdentity();

          Eigen::Quaterniond rot_h(p[nl_icp->S_[i].first * 7 + 3], p[nl_icp->S_[i].first * 7], p[nl_icp->S_[i].first * 7 + 1], p[nl_icp->S_[i].first * 7 + 2]);
          trans_h = Eigen::Vector3d(p[nl_icp->S_[i].first * 7 + 4], p[nl_icp->S_[i].first * 7 + 5], p[nl_icp->S_[i].first * 7 + 6]);

          Eigen::Quaterniond rot_k(p[nl_icp->S_[i].second * 7 + 3], p[nl_icp->S_[i].second * 7], p[nl_icp->S_[i].second * 7 + 1], p[nl_icp->S_[i].second * 7 + 2]);
          trans_k = Eigen::Vector3d (p[nl_icp->S_[i].second * 7 + 4], p[nl_icp->S_[i].second * 7 + 5], p[nl_icp->S_[i].second * 7 + 6]);

          R_h = coolquat_to_mat(rot_h);
          R_h /= rot_h.squaredNorm();
          R_k = coolquat_to_mat(rot_k);
          R_k /= rot_k.squaredNorm();

          Eigen::Matrix4f T;

          if(!MVNLICP_WEIRD_TRANS_)
          {
            T_h.block<3,3>(0,0) = R_h.cast<float>();
            T_h.block<3,1>(0,3) = trans_h.cast<float>();
            T_k.block<3,3>(0,0) = R_k.cast<float>();
            T_k.block<3,1>(0,3) = trans_k.cast<float>();
            T = T_k.inverse() * T_h;
          }

          //transform points from input cloud S_.first to S_.second reference frame
          //get correspondence of transformed point in S_.second distance transform

          for(size_t j=0; j < nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(); j++, k++)
          {
            PointT p;

            if(MVNLICP_WEIRD_TRANS_)
            {
              p.getVector3fMap() = R_h.cast<float>() * (nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector3fMap() + trans_h.cast<float>());
              p.getVector3fMap() = (R_k.cast<float>()).inverse() * (p.getVector3fMap() + (trans_k.cast<float>() * -1.f));
            }
            else
            {
              p.getVector4fMap() = T * nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector4fMap();
            }

            idx_match = -1;
            nl_icp->dist_transforms_[nl_icp->S_[i].second]->getCorrespondence(p, &idx_match, &distance, -1.f, &color_distance);
            if(distance > nl_icp->max_correspondence_distance_)
              continue;

            if((idx_match) >= 0)
            {
              used++;

              if(distance > nl_icp->max_correspondence_distance_)
              {
                x[k] = huberFitzigibon(nl_icp->max_correspondence_distance_, inliers_threshold_);
              }
              else
              {
                x[k] = huberFitzigibon(distance, inliers_threshold_);
              }
            }
            else
            {
              //x[k] = huberFitzigibon(nl_icp->max_correspondence_distance_, inliers_threshold_);
            }
          }

          total += nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size();
        }

        double sum=0;
        for(size_t k=0; k < n; k++)
        {
          sum += x[k] * x[k];
        }

        std::cout << "residuals sum:" << sum << " used:" << used << std::endl;
      }*/

      template <typename PointT>
      void model_func(double *p, double *x, int m, int n, void *data)
      {
        pcl::ScopeTime t("model_func");
        MVNonLinearICP<PointT> * nl_icp = (MVNonLinearICP<PointT> *)(data);

        /*double inliers_threshold_ = nl_icp->inlier_threshold_;
        double out = inliers_threshold_ * 1.5;

        for(size_t k=0; k < n; k++)
        {
          x[k] = huberFitzigibon(out, inliers_threshold_);
        }*/

        double inliers_threshold_ = nl_icp->inlier_threshold_ * 1.345;
        for(size_t k=0; k < n; k++)
          x[k] = huberFitzigibon(nl_icp->max_correspondence_distance_, inliers_threshold_);

        int k=0;
        int used=0;
        int total = 0;
        for(size_t i =0; i < nl_icp->S_.size(); i++)
        {
          int idx_match;
          float distance = 0;
          float color_distance = -1.f;

          Eigen::Matrix3d R_h, R_k;
          Eigen::Vector3d trans_h, trans_k;
          Eigen::Matrix4f T_h, T_k;
          T_h.setIdentity();
          T_k.setIdentity();

          Eigen::Quaterniond rot_h(p[nl_icp->S_[i].first * 7 + 3], p[nl_icp->S_[i].first * 7], p[nl_icp->S_[i].first * 7 + 1], p[nl_icp->S_[i].first * 7 + 2]);
          trans_h = Eigen::Vector3d(p[nl_icp->S_[i].first * 7 + 4], p[nl_icp->S_[i].first * 7 + 5], p[nl_icp->S_[i].first * 7 + 6]);

          Eigen::Quaterniond rot_k(p[nl_icp->S_[i].second * 7 + 3], p[nl_icp->S_[i].second * 7], p[nl_icp->S_[i].second * 7 + 1], p[nl_icp->S_[i].second * 7 + 2]);
          trans_k = Eigen::Vector3d (p[nl_icp->S_[i].second * 7 + 4], p[nl_icp->S_[i].second * 7 + 5], p[nl_icp->S_[i].second * 7 + 6]);

          R_h = coolquat_to_mat(rot_h);
          R_h /= rot_h.squaredNorm();
          R_k = coolquat_to_mat(rot_k);
          R_k /= rot_k.squaredNorm();

          Eigen::Matrix4f T;

          if(!MVNLICP_WEIRD_TRANS_)
          {
            T_h.block<3,3>(0,0) = R_h.cast<float>();
            T_h.block<3,1>(0,3) = trans_h.cast<float>();
            T_k.block<3,3>(0,0) = R_k.cast<float>();
            T_k.block<3,1>(0,3) = trans_k.cast<float>();
            T = T_k.inverse() * T_h;
          }

          //transform points from input cloud S_.first to S_.second reference frame
          //get correspondence of transformed point in S_.second distance transform

          std::vector<int> nneigh(nl_icp->input_clouds_[nl_icp->S_[i].second]->points.size(),0);
          std::vector<double> min_dist_nneigh(nl_icp->input_clouds_[nl_icp->S_[i].second]->points.size(),std::numeric_limits<double>::infinity());
          std::vector<int> inputcloud_to_target(nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(),-1);

          for(size_t j=0; j < nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(); j++)
          {
            PointT p;

            if(MVNLICP_WEIRD_TRANS_)
            {
              p.getVector3fMap() = R_h.cast<float>() * (nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector3fMap() + trans_h.cast<float>());
              p.getVector3fMap() = (R_k.cast<float>()).inverse() * (p.getVector3fMap() + (trans_k.cast<float>() * -1.f));
            }
            else
            {
              p.getVector4fMap() = T * nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector4fMap();
            }

            idx_match = -1;
            /*nl_icp->dist_transforms_[nl_icp->S_[i].second]->getCorrespondence(p, &idx_match, &distance, -1.f, &color_distance);
            if(distance > nl_icp->max_correspondence_distance_)
              continue;*/

            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            if (nl_icp->octrees_[nl_icp->S_[i].second]->nearestKSearch (p, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
              idx_match = pointIdxNKNSearch[0];
              distance = sqrt(pointNKNSquaredDistance[0]);
            }

            if((idx_match) >= 0)
            {
              /*if(nl_icp->normals_available_)
              {
                Eigen::Vector3f diff = p.getVector3fMap() - nl_icp->input_clouds_[nl_icp->S_[i].second]->points[idx_match].getVector3fMap();
                Eigen::Vector3f normal_target = nl_icp->input_normals_[nl_icp->S_[i].second]->points[idx_match].getNormalVector3fMap();
                normal_target.normalize();
                distance = std::abs(normal_target.dot(diff));
              }*/

              if(nl_icp->normals_available_)
              {
                Eigen::Vector3f p_normal = nl_icp->input_normals_[nl_icp->S_[i].first]->points[j].getNormalVector3fMap();
                Eigen::Vector3f match_normal = nl_icp->input_normals_[nl_icp->S_[i].second]->points[idx_match].getNormalVector3fMap();
                p_normal.normalize(); match_normal.normalize();
                if(p_normal.dot(match_normal) > nl_icp->min_dot_)
                {
                  inputcloud_to_target[j] = idx_match;
                  if(distance < min_dist_nneigh[idx_match])
                  {
                    min_dist_nneigh[idx_match] = distance;
                    nneigh[idx_match] = j;
                  }
                }
              }
              else
              {
                inputcloud_to_target[j] = idx_match;
                if(distance < min_dist_nneigh[idx_match])
                {
                  min_dist_nneigh[idx_match] = distance;
                  nneigh[idx_match] = j;
                }
              }

              /*inputcloud_to_target[j] = idx_match;

              if(distance < min_dist_nneigh[idx_match])
              {
                min_dist_nneigh[idx_match] = distance;
                nneigh[idx_match] = j;
              }*/

            }
          }

          for(size_t j=0; j < nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(); j++,k++)
          {

            if(inputcloud_to_target[j] == -1)
              continue;

            if(nneigh[inputcloud_to_target[j]] != j)
              continue;

            if(pcl_isfinite(min_dist_nneigh[inputcloud_to_target[j]]))
            {
              used++;
              float w = 1.f;
              if(nl_icp->weights_available_)
                w = (*(nl_icp->weights_))[nl_icp->S_[i].first][j] * (*(nl_icp->weights_))[nl_icp->S_[i].second][idx_match];

              x[k] = w * huberFitzigibon(min_dist_nneigh[inputcloud_to_target[j]], inliers_threshold_);
            }
          }


          total += nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size();
        }

        double sum=0;
        for(size_t k=0; k < n; k++)
        {
          sum += x[k] * x[k];
        }

        std::cout << "residuals sum:" << sum << " used:" << used << std::endl;
      }

      template <typename PointT>
      void jac_func(double *p, double *jac, int m, int n, void *data)
      {
        pcl::ScopeTime t("jac_func");
        MVNonLinearICP<PointT> * nl_icp = (MVNonLinearICP<PointT> *)(data);
        for(int k=0; k<n*m; ++k)
        {
          jac[k] = 0;
        }

        //fill jacobian row-block by row-block
        std::vector<int> rows;
        int row = 0;
        for(size_t i =0; i < nl_icp->S_.size(); i++)
        {
          rows.push_back(row);
          row += nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size();
        }

//#pragma omp parallel for num_threads(8)
        for(size_t i =0; i < nl_icp->S_.size(); i++)
        {

          Eigen::Matrix3d R_h, R_k;
          Eigen::Vector3d trans_h, trans_k;
          Eigen::Matrix4f T_h, T_k;
          T_h.setIdentity();
          T_k.setIdentity();

          Eigen::Quaterniond rot_h(p[nl_icp->S_[i].first * 7 + 3], p[nl_icp->S_[i].first * 7], p[nl_icp->S_[i].first * 7 + 1], p[nl_icp->S_[i].first * 7 + 2]);
          trans_h = Eigen::Vector3d(p[nl_icp->S_[i].first * 7 + 4], p[nl_icp->S_[i].first * 7 + 5], p[nl_icp->S_[i].first * 7 + 6]);

          Eigen::Quaterniond rot_k(p[nl_icp->S_[i].second * 7 + 3], p[nl_icp->S_[i].second * 7], p[nl_icp->S_[i].second * 7 + 1], p[nl_icp->S_[i].second * 7 + 2]);
          trans_k = Eigen::Vector3d (p[nl_icp->S_[i].second * 7 + 4], p[nl_icp->S_[i].second * 7 + 5], p[nl_icp->S_[i].second * 7 + 6]);

          R_h = coolquat_to_mat(rot_h);
          R_h /= rot_h.squaredNorm();
          R_k = coolquat_to_mat(rot_k);
          R_k /= rot_k.squaredNorm();

          Eigen::Matrix4f T;

          if(!MVNLICP_WEIRD_TRANS_)
          {
            T_h.block<3,3>(0,0) = R_h.cast<float>();
            T_h.block<3,1>(0,3) = trans_h.cast<float>();
            T_k.block<3,3>(0,0) = R_k.cast<float>();
            T_k.block<3,1>(0,3) = trans_k.cast<float>();
            T = T_k.inverse() * T_h;
            //T = T_h * T_k.inverse();
          }

          //fill block row by row
          int col_h, col_k;
          col_h = nl_icp->S_[i].first * 7;
          col_k = nl_icp->S_[i].second * 7;

          float x,y,z;
          float Lx,Ly,Lz;
          float qxh,qyh,qzh,qrh,txh,tyh,tzh;
          float qxk,qyk,qzk,qrk,txk,tyk,tzk;

          qxh = p[nl_icp->S_[i].first * 7 + 0];
          qyh = p[nl_icp->S_[i].first * 7 + 1];
          qzh = p[nl_icp->S_[i].first * 7 + 2];
          qrh = p[nl_icp->S_[i].first * 7 + 3];
          txh = p[nl_icp->S_[i].first * 7 + 4];
          tyh = p[nl_icp->S_[i].first * 7 + 5];
          tzh = p[nl_icp->S_[i].first * 7 + 6];

          qxk = p[nl_icp->S_[i].second * 7 + 0];
          qyk = p[nl_icp->S_[i].second * 7 + 1];
          qzk = p[nl_icp->S_[i].second * 7 + 2];
          qrk = p[nl_icp->S_[i].second * 7 + 3];
          txk = p[nl_icp->S_[i].second * 7 + 4];
          tyk = p[nl_icp->S_[i].second * 7 + 5];
          tzk = p[nl_icp->S_[i].second * 7 + 6];

          std::vector<int> nneigh(nl_icp->input_clouds_[nl_icp->S_[i].second]->points.size(),0);
          std::vector<double> min_dist_nneigh(nl_icp->input_clouds_[nl_icp->S_[i].second]->points.size(),std::numeric_limits<double>::infinity());
          std::vector<int> inputcloud_to_target(nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(),-1);

          for(size_t j=0; j < nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(); j++)
          {
            PointT p;
            if(MVNLICP_WEIRD_TRANS_)
            {
              p.getVector3fMap() = R_h.cast<float>() * (nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector3fMap() + trans_h.cast<float>());
              p.getVector3fMap() = (R_k.cast<float>()).inverse() * (p.getVector3fMap() + (trans_k.cast<float>() * -1.f));
            }
            else
            {
              p.getVector4fMap() = T * nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector4fMap();
            }

            int idx_match;
            float distance = 0;
            float color_distance = -1.f;
            nl_icp->dist_transforms_[nl_icp->S_[i].second]->getCorrespondence(p, &idx_match, &distance, -1.f, &color_distance);
            if(distance > nl_icp->max_correspondence_distance_)
              continue;

            inputcloud_to_target[j] = idx_match;

            if(distance < min_dist_nneigh[idx_match])
            {
              min_dist_nneigh[idx_match] = distance;
              nneigh[idx_match] = j;
            }
          }

          for(size_t j=0; j < nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(); j++)
          {
            if(inputcloud_to_target[j] == -1)
              continue;

            if(nneigh[inputcloud_to_target[j]] != j)
              continue;

            PointT p;
            if(MVNLICP_WEIRD_TRANS_)
            {
              p.getVector3fMap() = R_h.cast<float>() * (nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector3fMap() + trans_h.cast<float>());
              p.getVector3fMap() = (R_k.cast<float>()).inverse() * (p.getVector3fMap() + (trans_k.cast<float>() * -1.f));
            }
            else
            {
              p.getVector4fMap() = T * nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector4fMap();
            }

            Eigen::Vector3f L;
            nl_icp->dist_transforms_[nl_icp->S_[i].second]->getDerivatives(p, L);

            Lx = L[0];
            Ly = L[1];
            Lz = L[2];

            x = p.x;
            y = p.y;
            z = p.z;

            if(pcl_isnan(Lx) || pcl_isnan(Ly) || pcl_isnan(Lz))
            {
              //PCL_WARN("Derivatives are NaN...\n");
              continue;
            }

            int pos = (rows[i] + j) * m;
            if(MVNLICP_WEIRD_TRANS_)
            {
              jac[pos + col_h]= -Lx*(-((qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qxh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qxh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qzk-qxk*qyk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qxh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qyk+qxk*qzk)*((qzh*(txh+x)*-2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qxh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qxh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+Ly*(((qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qxh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qxk-qyk*qzk)*((qzh*(txh+x)*-2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qxh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qxh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qzk+qxk*qyk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qxh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qxh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-Lz*(((qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*((qzh*(txh+x)*-2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qxh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qxh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-((qrk*qxk+qyk*qzk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qxh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qyk-qxk*qzk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qxh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qxh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qxh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk));
              jac[pos + col_h + 1]= -Lx*(((qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qyh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qyh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-((qrk*qyk+qxk*qzk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qyh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qyh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qzk-qxk*qyk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qyh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-Ly*(-((qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qyh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qzk+qxk*qyk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qyh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qyh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qxk-qyk*qzk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qyh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qyh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+Lz*(((qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qyh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qyh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qyk-qxk*qzk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qyh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qyh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qxk+qyk*qzk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qyh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qyh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk));
              jac[pos + col_h + 2]= Lx*(-((qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*((qzh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qzh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qzh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qzk-qxk*qyk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qzh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qyk+qxk*qzk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qzh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qzh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-Ly*(((qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qzh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qzk+qxk*qyk)*((qzh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qzh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qzh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qxk-qyk*qzk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qzh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qzh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+Lz*(((qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*((qxh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qzh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qzh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qyk-qxk*qzk)*((qzh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+qzh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qzh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-((qrk*qxk+qyk*qzk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qzh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qzh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk));
              jac[pos + col_h + 3]= Lx*(((qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qrh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qyk+qxk*qzk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qrh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qzk-qxk*qyk)*((qzh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-Ly*(((qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*((qzh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qxk-qyk*qzk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qrh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-((qrk*qzk+qxk*qyk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qrh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-Lz*(-((qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*((qyh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qrh*(tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qxk+qyk*qzk)*((qzh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrk*qyk-qxk*qzk)*((qrh*(txh+x)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzh*(tyh+y)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyh*(tzh+z)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-qrh*(tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0+qrh*(tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0-qrh*(txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk));
              jac[pos + col_h + 4]= Lx*((((qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+((qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))-Ly*((((qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qzk+qxk*qyk)*-2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+((qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk))/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))-Lz*((((qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qyk-qxk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-((qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk))/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
              jac[pos + col_h + 5]= -Lx*((((qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qzk-qxk*qyk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-((qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk))/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Ly*((((qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+((qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))-Lz*((((qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qxk+qyk*qzk)*-2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+((qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk))/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
              jac[pos + col_h + 6]= -Lx*((((qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qyk+qxk*qzk)*-2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+((qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk))/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))-Ly*((((qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qxk-qyk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-((qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk))/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Lz*((((qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-1.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+((qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+((qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));

              jac[pos + col_k]= Ly*((qrk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qxk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0+qxk*(qrk*qzk+qxk*qyk)*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*(qrk*qxk-qyk*qzk)*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)-Lz*((qrk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qxk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0+qxk*(qrk*qyk-qxk*qzk)*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*(qrk*qxk+qyk*qzk)*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)-Lx*((qxk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qxk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0+qxk*(qrk*qzk-qxk*qyk)*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*(qrk*qyk+qxk*qzk)*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0);
              jac[pos + col_k + 1]= -Lx*((qrk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qyk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0+qyk*(qrk*qzk-qxk*qyk)*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qyk*(qrk*qyk+qxk*qzk)*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)+Lz*((qrk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qyk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0-qyk*(qrk*qyk-qxk*qzk)*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qyk*(qrk*qxk+qyk*qzk)*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)-Ly*((qxk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qyk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0-qyk*(qrk*qzk+qxk*qyk)*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qyk*(qrk*qxk-qyk*qzk)*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0);
              jac[pos + col_k + 2]= Lx*((qrk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qzk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0-qzk*(qrk*qzk-qxk*qyk)*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qzk*(qrk*qyk+qxk*qzk)*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)-Ly*((qrk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qzk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0-qzk*(qrk*qzk+qxk*qyk)*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qzk*(qrk*qxk-qyk*qzk)*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)-Lz*((qxk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qzk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0+qzk*(qrk*qyk-qxk*qzk)*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qzk*(qrk*qxk+qyk*qzk)*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0);
              jac[pos + col_k + 3]= -Lx*((qrk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qrk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0+qrk*(qrk*qzk-qxk*qyk)*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*(qrk*qyk+qxk*qzk)*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)-Ly*((qrk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qrk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0-qrk*(qrk*qzk+qxk*qyk)*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qrk*(qrk*qxk-qyk*qzk)*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)-Lz*((qrk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qrk*(tzk-((tzh+z)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((txh+x)*(qrh*qyh*2.0+qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tyh+y)*(qrh*qxh*2.0-qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0+qrk*(qrk*qyk-qxk*qzk)*(txk-((txh+x)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tyh+y)*(qrh*qzh*2.0+qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((tzh+z)*(qrh*qyh*2.0-qxh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*(qrk*qxk+qyk*qzk)*(tyk-((tyh+y)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+((txh+x)*(qrh*qzh*2.0-qxh*qyh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-((tzh+z)*(qrh*qxh*2.0+qyh*qzh*2.0))/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0);
              jac[pos + col_k + 4]= -(Lx*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Ly*(qrk*qzk+qxk*qyk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lz*(qrk*qyk-qxk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
              jac[pos + col_k + 5]= -(Ly*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lx*(qrk*qzk-qxk*qyk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lz*(qrk*qxk+qyk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
              jac[pos + col_k + 6]= -(Lz*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lx*(qrk*qyk+qxk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Ly*(qrk*qxk-qyk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
            }
            else
            {
              jac[pos + col_h] = Lx*(-x*((qxh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qyh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qzh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Ly*(x*((qyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-y*((qxh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qrh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))-Lz*(-x*((qzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qrh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qxh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
              jac[pos + col_h + 1] = -Lx*(x*((qyh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-y*((qxh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qrh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Ly*(x*((qxh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-y*((qyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qzh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Lz*(x*((qrh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-z*((qyh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
              jac[pos + col_h + 2] = Lx*(-x*((qzh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qrh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qxh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))-Ly*(x*((qrh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qzh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-z*((qyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Lz*(x*((qxh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qyh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-z*((qzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
              jac[pos + col_h + 3] = Lx*(x*((qrh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qzh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qrh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-z*((qyh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Ly*(-x*((qzh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qrh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qxh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qrh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Lz*(x*((qyh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qrh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-y*((qxh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qrh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
              jac[pos + col_h + 4] = (Lx*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Ly*(qrk*qzk+qxk*qyk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lz*(qrk*qyk-qxk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
              jac[pos + col_h + 5] = (Ly*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lx*(qrk*qzk-qxk*qyk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lz*(qrk*qxk+qyk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
              jac[pos + col_h + 6] = (Lz*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lx*(qrk*qyk+qxk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Ly*(qrk*qxk-qyk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);

              jac[pos + col_k] = -Ly*(-(qrk*tzk*2.0+qxk*tyk*2.0-qyk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-x*((qyk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qxk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qrk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qxk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tyk-(qxk*qxk)*tyk+(qyk*qyk)*tyk-(qzk*qzk)*tyk-qrk*qxk*tzk*2.0+qrk*qzk*txk*2.0+qxk*qyk*txk*2.0+qyk*qzk*tzk*2.0)*2.0+qxk*txh*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*tzh*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qxk*tyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Lz*(-(qrk*tyk*2.0-qxk*tzk*2.0+qzk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qzk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qrk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-z*((qxk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qxk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tzk-(qxk*qxk)*tzk-(qyk*qyk)*tzk+(qzk*qzk)*tzk+qrk*qxk*tyk*2.0-qrk*qyk*txk*2.0+qxk*qzk*txk*2.0+qyk*qzk*tyk*2.0)*2.0+qxk*txh*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*tyh*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*tzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Lx*(-(qxk*txk*2.0+qyk*tyk*2.0+qzk*tzk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-x*((qxk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qyk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qzk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qxk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qxk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*txk+(qxk*qxk)*txk-(qyk*qyk)*txk-(qzk*qzk)*txk+qrk*qyk*tzk*2.0-qrk*qzk*tyk*2.0+qxk*qyk*tyk*2.0+qxk*qzk*tzk*2.0)*2.0+qxk*tyh*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*tzh*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*txh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0);
              jac[pos + col_k + 1] = Lx*(-(qrk*tzk*2.0+qxk*tyk*2.0-qyk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-x*((qyk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qxk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qrk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qyk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*txk+(qxk*qxk)*txk-(qyk*qyk)*txk-(qzk*qzk)*txk+qrk*qyk*tzk*2.0-qrk*qzk*tyk*2.0+qxk*qyk*tyk*2.0+qxk*qzk*tzk*2.0)*2.0+qyk*tyh*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qyk*tzh*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qyk*txh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)-Lz*(-(qrk*txk*2.0+qyk*tzk*2.0-qzk*tyk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qrk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-y*((qzk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qyk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qyk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tzk-(qxk*qxk)*tzk-(qyk*qyk)*tzk+(qzk*qzk)*tzk+qrk*qxk*tyk*2.0-qrk*qyk*txk*2.0+qxk*qzk*txk*2.0+qyk*qzk*tyk*2.0)*2.0-qyk*txh*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qyk*tyh*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qyk*tzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Ly*(-(qxk*txk*2.0+qyk*tyk*2.0+qzk*tzk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qxk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-y*((qyk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qzk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qxk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qyk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tyk-(qxk*qxk)*tyk+(qyk*qyk)*tyk-(qzk*qzk)*tyk-qrk*qxk*tzk*2.0+qrk*qzk*txk*2.0+qxk*qyk*txk*2.0+qyk*qzk*tzk*2.0)*2.0-qyk*txh*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qyk*tzh*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qyk*tyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0);
              jac[pos + col_k + 2] = -Lx*(-(qrk*tyk*2.0-qxk*tzk*2.0+qzk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qzk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qrk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-z*((qxk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qzk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*txk+(qxk*qxk)*txk-(qyk*qyk)*txk-(qzk*qzk)*txk+qrk*qyk*tzk*2.0-qrk*qzk*tyk*2.0+qxk*qyk*tyk*2.0+qxk*qzk*tzk*2.0)*2.0-qzk*tyh*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qzk*tzh*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qzk*txh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Ly*(-(qrk*txk*2.0+qyk*tzk*2.0-qzk*tyk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qrk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-y*((qzk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qyk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qzk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tyk-(qxk*qxk)*tyk+(qyk*qyk)*tyk-(qzk*qzk)*tyk-qrk*qxk*tzk*2.0+qrk*qzk*txk*2.0+qxk*qyk*txk*2.0+qyk*qzk*tzk*2.0)*2.0-qzk*txh*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qzk*tzh*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qzk*tyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Lz*(-(qxk*txk*2.0+qyk*tyk*2.0+qzk*tzk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qxk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qyk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-z*((qzk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qxk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qzk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tzk-(qxk*qxk)*tzk-(qyk*qyk)*tzk+(qzk*qzk)*tzk+qrk*qxk*tyk*2.0-qrk*qyk*txk*2.0+qxk*qzk*txk*2.0+qyk*qzk*tyk*2.0)*2.0+qzk*txh*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qzk*tyh*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qzk*tzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0);
              jac[pos + col_k + 3] = Lz*(-(qrk*tzk*2.0+qxk*tyk*2.0-qyk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-x*((qyk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qxk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qrk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qrk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tzk-(qxk*qxk)*tzk-(qyk*qyk)*tzk+(qzk*qzk)*tzk+qrk*qxk*tyk*2.0-qrk*qyk*txk*2.0+qxk*qzk*txk*2.0+qyk*qzk*tyk*2.0)*2.0+qrk*txh*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*tyh*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*tzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Ly*(-(qrk*tyk*2.0-qxk*tzk*2.0+qzk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qzk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qrk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-z*((qxk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qrk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tyk-(qxk*qxk)*tyk+(qyk*qyk)*tyk-(qzk*qzk)*tyk-qrk*qxk*tzk*2.0+qrk*qzk*txk*2.0+qxk*qyk*txk*2.0+qyk*qzk*tzk*2.0)*2.0-qrk*txh*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qrk*tzh*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*tyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Lx*(-(qrk*txk*2.0+qyk*tzk*2.0-qzk*tyk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qrk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-y*((qzk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qyk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qrk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*txk+(qxk*qxk)*txk-(qyk*qyk)*txk-(qzk*qzk)*txk+qrk*qyk*tzk*2.0-qrk*qzk*tyk*2.0+qxk*qyk*tyk*2.0+qxk*qzk*tzk*2.0)*2.0+qrk*tyh*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*tzh*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*txh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0);
              jac[pos + col_k + 4] = -(Lx*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Ly*(qrk*qzk*2.0+qxk*qyk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lz*(qrk*qyk*2.0-qxk*qzk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
              jac[pos + col_k + 5] = -(Ly*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lx*(qrk*qzk*2.0-qxk*qyk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lz*(qrk*qxk*2.0+qyk*qzk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
              jac[pos + col_k + 6] = -(Lz*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lx*(qrk*qyk*2.0+qxk*qzk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Ly*(qrk*qxk*2.0-qyk*qzk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
            }
          }

          //row += nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size();
        }
      }

      template <typename PointT>
      void jac_func_crs(double *p, struct splm_ccsm *jac, int m, int n, void *data)
      {
        pcl::ScopeTime t("jac_func_crs");
        struct splm_stm jac_st;

        //allocate and fill-in a ST Jacobian...
        splm_stm_allocval(&jac_st, n, m, jac->nnz);
        MVNonLinearICP<PointT> * nl_icp = (MVNonLinearICP<PointT> *)(data);
        double huber_sigma = nl_icp->inlier_threshold_ * 1.345;

        //fill jacobian row-block by row-block
        std::vector<int> rows;
        int row = 0;
        for(size_t i =0; i < nl_icp->S_.size(); i++)
        {
          rows.push_back(row);
          row += nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size();
        }

#pragma omp parallel for num_threads(8)
        for(size_t i =0; i < nl_icp->S_.size(); i++)
        {

          Eigen::Matrix3d R_h, R_k;
          Eigen::Vector3d trans_h, trans_k;
          Eigen::Matrix4f T_h, T_k;
          T_h.setIdentity();
          T_k.setIdentity();

          Eigen::Quaterniond rot_h(p[nl_icp->S_[i].first * 7 + 3], p[nl_icp->S_[i].first * 7], p[nl_icp->S_[i].first * 7 + 1], p[nl_icp->S_[i].first * 7 + 2]);
          trans_h = Eigen::Vector3d(p[nl_icp->S_[i].first * 7 + 4], p[nl_icp->S_[i].first * 7 + 5], p[nl_icp->S_[i].first * 7 + 6]);

          Eigen::Quaterniond rot_k(p[nl_icp->S_[i].second * 7 + 3], p[nl_icp->S_[i].second * 7], p[nl_icp->S_[i].second * 7 + 1], p[nl_icp->S_[i].second * 7 + 2]);
          trans_k = Eigen::Vector3d (p[nl_icp->S_[i].second * 7 + 4], p[nl_icp->S_[i].second * 7 + 5], p[nl_icp->S_[i].second * 7 + 6]);

          R_h = coolquat_to_mat(rot_h);
          R_h /= rot_h.squaredNorm();
          R_k = coolquat_to_mat(rot_k);
          R_k /= rot_k.squaredNorm();

          Eigen::Matrix4f T;

          if(!MVNLICP_WEIRD_TRANS_)
          {
            T_h.block<3,3>(0,0) = R_h.cast<float>();
            T_h.block<3,1>(0,3) = trans_h.cast<float>();
            T_k.block<3,3>(0,0) = R_k.cast<float>();
            T_k.block<3,1>(0,3) = trans_k.cast<float>();
            T = T_k.inverse() * T_h;
            //T = T_h * T_k.inverse();
          }

          //fill block row by row
          int col_h, col_k;
          col_h = nl_icp->S_[i].first * 7;
          col_k = nl_icp->S_[i].second * 7;

          float x,y,z;
          float Lx,Ly,Lz;
          float qxh,qyh,qzh,qrh,txh,tyh,tzh;
          float qxk,qyk,qzk,qrk,txk,tyk,tzk;

          qxh = p[nl_icp->S_[i].first * 7 + 0];
          qyh = p[nl_icp->S_[i].first * 7 + 1];
          qzh = p[nl_icp->S_[i].first * 7 + 2];
          qrh = p[nl_icp->S_[i].first * 7 + 3];
          txh = p[nl_icp->S_[i].first * 7 + 4];
          tyh = p[nl_icp->S_[i].first * 7 + 5];
          tzh = p[nl_icp->S_[i].first * 7 + 6];

          qxk = p[nl_icp->S_[i].second * 7 + 0];
          qyk = p[nl_icp->S_[i].second * 7 + 1];
          qzk = p[nl_icp->S_[i].second * 7 + 2];
          qrk = p[nl_icp->S_[i].second * 7 + 3];
          txk = p[nl_icp->S_[i].second * 7 + 4];
          tyk = p[nl_icp->S_[i].second * 7 + 5];
          tzk = p[nl_icp->S_[i].second * 7 + 6];

          std::vector<int> nneigh(nl_icp->input_clouds_[nl_icp->S_[i].second]->points.size(),0);
          std::vector<double> min_dist_nneigh(nl_icp->input_clouds_[nl_icp->S_[i].second]->points.size(),std::numeric_limits<double>::infinity());
          std::vector<int> inputcloud_to_target(nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(),-1);

          for(size_t j=0; j < nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(); j++)
          {
            PointT p;
            if(MVNLICP_WEIRD_TRANS_)
            {
              p.getVector3fMap() = R_h.cast<float>() * (nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector3fMap() + trans_h.cast<float>());
              p.getVector3fMap() = (R_k.cast<float>()).inverse() * (p.getVector3fMap() + (trans_k.cast<float>() * -1.f));
            }
            else
            {
              p.getVector4fMap() = T * nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector4fMap();
            }

            int idx_match;
            float distance = 0;

            /*float color_distance = -1.f;
            nl_icp->dist_transforms_[nl_icp->S_[i].second]->getCorrespondence(p, &idx_match, &distance, -1.f, &color_distance);*/

            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            if (nl_icp->octrees_[nl_icp->S_[i].second]->nearestKSearch (p, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
              idx_match = pointIdxNKNSearch[0];
              distance = sqrt(pointNKNSquaredDistance[0]);
            }

            if(distance > nl_icp->max_correspondence_distance_)
              continue;

            //check if the normals are ok to avoid matching opposing surfaces
            if(nl_icp->normals_available_)
            {
              Eigen::Vector3f p_normal = nl_icp->input_normals_[nl_icp->S_[i].first]->points[j].getNormalVector3fMap();
              Eigen::Vector3f match_normal = nl_icp->input_normals_[nl_icp->S_[i].second]->points[idx_match].getNormalVector3fMap();
              p_normal.normalize(); match_normal.normalize();
              if(p_normal.dot(match_normal) > nl_icp->min_dot_)
              {
                inputcloud_to_target[j] = idx_match;
                if(distance < min_dist_nneigh[idx_match])
                {
                  min_dist_nneigh[idx_match] = distance;
                  nneigh[idx_match] = j;
                }
              }
            }
            else
            {
              inputcloud_to_target[j] = idx_match;
              if(distance < min_dist_nneigh[idx_match])
              {
                min_dist_nneigh[idx_match] = distance;
                nneigh[idx_match] = j;
              }
            }
          }

          for(size_t j=0; j < nl_icp->input_clouds_[nl_icp->S_[i].first]->points.size(); j++)
          {
            if(inputcloud_to_target[j] == -1)
              continue;

            if(nneigh[inputcloud_to_target[j]] != j)
              continue;

            PointT p;
            if(MVNLICP_WEIRD_TRANS_)
            {
              p.getVector3fMap() = R_h.cast<float>() * (nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector3fMap() + trans_h.cast<float>());
              p.getVector3fMap() = (R_k.cast<float>()).inverse() * (p.getVector3fMap() + (trans_k.cast<float>() * -1.f));
            }
            else
            {
              p.getVector4fMap() = T * nl_icp->input_clouds_[nl_icp->S_[i].first]->points[j].getVector4fMap();
            }

            Eigen::Vector3f L;
            float step = 0.002f;

            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;

            PointT p_plus, p_minus;
            int idx_match_plus, idx_match_minus;
            float distance_plus, distance_minus;

            { //X
              p_plus = p;
              p_minus = p;
              p_plus.x += step / 2.f;
              p_minus.x -= step / 2.f;

              if (nl_icp->octrees_[nl_icp->S_[i].second]->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
              {
                idx_match_plus = pointIdxNKNSearch[0];
                distance_plus = sqrt(pointNKNSquaredDistance[0]);
              }

              if (nl_icp->octrees_[nl_icp->S_[i].second]->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
              {
                idx_match_minus = pointIdxNKNSearch[0];
                distance_minus = sqrt(pointNKNSquaredDistance[0]);
              }

              Lx = sqrt(huberFitzigibon(distance_plus, huber_sigma)) - sqrt(huberFitzigibon(distance_minus, huber_sigma));
            }

            { //Y
              p_plus = p;
              p_minus = p;
              p_plus.y += step / 2.f;
              p_minus.y -= step / 2.f;

              if (nl_icp->octrees_[nl_icp->S_[i].second]->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
              {
                idx_match_plus = pointIdxNKNSearch[0];
                distance_plus = sqrt(pointNKNSquaredDistance[0]);
              }

              if (nl_icp->octrees_[nl_icp->S_[i].second]->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
              {
                idx_match_minus = pointIdxNKNSearch[0];
                distance_minus = sqrt(pointNKNSquaredDistance[0]);
              }

              Ly = sqrt(huberFitzigibon(distance_plus, huber_sigma)) - sqrt(huberFitzigibon(distance_minus, huber_sigma));
            }

            { //Z
              p_plus = p;
              p_minus = p;
              p_plus.z += step / 2.f;
              p_minus.z -= step / 2.f;

              if (nl_icp->octrees_[nl_icp->S_[i].second]->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
              {
                idx_match_plus = pointIdxNKNSearch[0];
                distance_plus = sqrt(pointNKNSquaredDistance[0]);
              }

              if (nl_icp->octrees_[nl_icp->S_[i].second]->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
              {
                idx_match_minus = pointIdxNKNSearch[0];
                distance_minus = sqrt(pointNKNSquaredDistance[0]);
              }

              Lz = sqrt(huberFitzigibon(distance_plus, huber_sigma)) - sqrt(huberFitzigibon(distance_minus, huber_sigma));
            }
            /*nl_icp->dist_transforms_[nl_icp->S_[i].second]->getDerivatives(p, L);

            Lx = L[0];
            Ly = L[1];
            Lz = L[2];*/

            x = p.x;
            y = p.y;
            z = p.z;

            if(pcl_isnan(Lx) || pcl_isnan(Ly) || pcl_isnan(Lz))
            {
              //PCL_WARN("Derivatives are NaN...\n");
              continue;
            }

            int pos = (rows[i] + j) * m;
            assert((rows[i] + j) < n);
            assert((col_h + 6) < m);
            assert((col_k + 6) < m);

            float w = 1.f;
            if(nl_icp->weights_available_)
              w = (*(nl_icp->weights_))[nl_icp->S_[i].first][j] * (*(nl_icp->weights_))[nl_icp->S_[i].second][inputcloud_to_target[j]];

            /*if(nl_icp->weights_available_)
              w = (*(nl_icp->weights_))[nl_icp->S_[i].first][j];*/

            float j1 = w * Lx*(-x*((qxh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qyh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qzh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Ly*(x*((qyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-y*((qxh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qrh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))-Lz*(-x*((qzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qrh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qxh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
            float j2 = w * -Lx*(x*((qyh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-y*((qxh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qrh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Ly*(x*((qxh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-y*((qyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qzh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Lz*(x*((qrh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-z*((qyh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
            float j3 = w * Lx*(-x*((qzh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qrh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qxh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))-Ly*(x*((qrh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qzh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-z*((qyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Lz*(x*((qxh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qyh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-z*((qzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
            float j4 = w * Lx*(x*((qrh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qzh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qrh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-z*((qyh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk+qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qzk-qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Ly*(-x*((qzh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+y*((qrh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qxh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk-qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyh*(qrk*qzk+qxk*qyk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qrh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)))+Lz*(x*((qyh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qrh*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-y*((qxh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qrh*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+z*((qrh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxh*(qrk*qxk+qyk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyh*(qrk*qyk-qxk*qzk)*4.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrh*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qrh*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh,2.0)*4.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)));
            float j5 = w * (Lx*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Ly*(qrk*qzk+qxk*qyk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lz*(qrk*qyk-qxk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
            float j6 = w * (Ly*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lx*(qrk*qzk-qxk*qyk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lz*(qrk*qxk+qyk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
            float j7 = w * (Lz*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lx*(qrk*qyk+qxk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Ly*(qrk*qxk-qyk*qzk)*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);

            float j8 = w * -Ly*(-(qrk*tzk*2.0+qxk*tyk*2.0-qyk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-x*((qyk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qxk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qrk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qxk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tyk-(qxk*qxk)*tyk+(qyk*qyk)*tyk-(qzk*qzk)*tyk-qrk*qxk*tzk*2.0+qrk*qzk*txk*2.0+qxk*qyk*txk*2.0+qyk*qzk*tzk*2.0)*2.0+qxk*txh*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*tzh*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qxk*tyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Lz*(-(qrk*tyk*2.0-qxk*tzk*2.0+qzk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qzk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qrk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-z*((qxk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qxk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tzk-(qxk*qxk)*tzk-(qyk*qyk)*tzk+(qzk*qzk)*tzk+qrk*qxk*tyk*2.0-qrk*qyk*txk*2.0+qxk*qzk*txk*2.0+qyk*qzk*tyk*2.0)*2.0+qxk*txh*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*tyh*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*tzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Lx*(-(qxk*txk*2.0+qyk*tyk*2.0+qzk*tzk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-x*((qxk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qyk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qzk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qxk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qxk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*txk+(qxk*qxk)*txk-(qyk*qyk)*txk-(qzk*qzk)*txk+qrk*qyk*tzk*2.0-qrk*qzk*tyk*2.0+qxk*qyk*tyk*2.0+qxk*qzk*tzk*2.0)*2.0+qxk*tyh*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*tzh*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qxk*txh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0);
            float j9 = w * Lx*(-(qrk*tzk*2.0+qxk*tyk*2.0-qyk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-x*((qyk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qxk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qrk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qyk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*txk+(qxk*qxk)*txk-(qyk*qyk)*txk-(qzk*qzk)*txk+qrk*qyk*tzk*2.0-qrk*qzk*tyk*2.0+qxk*qyk*tyk*2.0+qxk*qzk*tzk*2.0)*2.0+qyk*tyh*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qyk*tzh*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qyk*txh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)-Lz*(-(qrk*txk*2.0+qyk*tzk*2.0-qzk*tyk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qrk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-y*((qzk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qyk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qyk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tzk-(qxk*qxk)*tzk-(qyk*qyk)*tzk+(qzk*qzk)*tzk+qrk*qxk*tyk*2.0-qrk*qyk*txk*2.0+qxk*qzk*txk*2.0+qyk*qzk*tyk*2.0)*2.0-qyk*txh*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qyk*tyh*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qyk*tzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Ly*(-(qxk*txk*2.0+qyk*tyk*2.0+qzk*tzk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qxk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-y*((qyk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qzk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qxk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qyk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tyk-(qxk*qxk)*tyk+(qyk*qyk)*tyk-(qzk*qzk)*tyk-qrk*qxk*tzk*2.0+qrk*qzk*txk*2.0+qxk*qyk*txk*2.0+qyk*qzk*tzk*2.0)*2.0-qyk*txh*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qyk*tzh*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qyk*tyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0);
            float j10 = w * -Lx*(-(qrk*tyk*2.0-qxk*tzk*2.0+qzk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qzk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qrk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-z*((qxk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-qzk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*txk+(qxk*qxk)*txk-(qyk*qyk)*txk-(qzk*qzk)*txk+qrk*qyk*tzk*2.0-qrk*qzk*tyk*2.0+qxk*qyk*tyk*2.0+qxk*qzk*tzk*2.0)*2.0-qzk*tyh*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qzk*tzh*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qzk*txh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Ly*(-(qrk*txk*2.0+qyk*tzk*2.0-qzk*tyk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qrk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-y*((qzk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qyk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qzk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tyk-(qxk*qxk)*tyk+(qyk*qyk)*tyk-(qzk*qzk)*tyk-qrk*qxk*tzk*2.0+qrk*qzk*txk*2.0+qxk*qyk*txk*2.0+qyk*qzk*tzk*2.0)*2.0-qzk*txh*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qzk*tzh*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qzk*tyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Lz*(-(qxk*txk*2.0+qyk*tyk*2.0+qzk*tzk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qxk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qzk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qyk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-z*((qzk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*-2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qxk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qzk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tzk-(qxk*qxk)*tzk-(qyk*qyk)*tzk+(qzk*qzk)*tzk+qrk*qxk*tyk*2.0-qrk*qyk*txk*2.0+qxk*qzk*txk*2.0+qyk*qzk*tyk*2.0)*2.0+qzk*txh*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qzk*tyh*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qzk*tzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0);
            float j11 = w * Lz*(-(qrk*tzk*2.0+qxk*tyk*2.0-qyk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-x*((qyk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qyk-qxk*qzk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qxk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qyk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qxk+qyk*qzk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qrk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qxk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qyk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qrk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tzk-(qxk*qxk)*tzk-(qyk*qyk)*tzk+(qzk*qzk)*tzk+qrk*qxk*tyk*2.0-qrk*qyk*txk*2.0+qxk*qzk*txk*2.0+qyk*qzk*tyk*2.0)*2.0+qrk*txh*(qrk*qyk-qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*tyh*(qrk*qxk+qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*tzh*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Ly*(-(qrk*tyk*2.0-qxk*tzk*2.0+qzk*txk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qzk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qxk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qzk+qxk*qyk)*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+y*((qrk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qxk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-z*((qxk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qxk-qyk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qxk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qzk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qrk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*tyk-(qxk*qxk)*tyk+(qyk*qyk)*tyk-(qzk*qzk)*tyk-qrk*qxk*tzk*2.0+qrk*qzk*txk*2.0+qxk*qyk*txk*2.0+qyk*qzk*tzk*2.0)*2.0-qrk*txh*(qrk*qzk+qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0+qrk*tzh*(qrk*qxk-qyk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*tyh*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)+Lx*(-(qrk*txk*2.0+qyk*tzk*2.0-qzk*tyk*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+x*((qrk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qyh*2.0+qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qzk*(qrh*qzh*2.0-qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qrh+qxh*qxh-qyh*qyh-qzh*qzh)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qyh*2.0+qxh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qzh*2.0-qxh*qyh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))-y*((qzk*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))+(qyk*(qrh*qxh*2.0-qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qzk-qxk*qyk)*(qrh*qrh-qxh*qxh+qyh*qyh-qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qzh*2.0+qxh*qyh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)-(qrk*(qrh*qxh*2.0-qyh*qzh*2.0)*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+z*((qyk*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qzk*(qrh*qxh*2.0+qyh*qzh*2.0)*2.0)/((qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)*(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk))-(qrk*(qrk*qyk+qxk*qzk)*(qrh*qrh-qxh*qxh-qyh*qyh+qzh*qzh)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qyh*2.0-qxh*qzh*2.0)*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh)+(qrk*(qrh*qxh*2.0+qyh*qzh*2.0)*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0)/(qrh*qrh+qxh*qxh+qyh*qyh+qzh*qzh))+(qrk*txh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(qyk*tzh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(qzk*tyh*2.0)/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+qrk*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*((qrk*qrk)*txk+(qxk*qxk)*txk-(qyk*qyk)*txk-(qzk*qzk)*txk+qrk*qyk*tzk*2.0-qrk*qzk*tyk*2.0+qxk*qyk*tyk*2.0+qxk*qzk*tzk*2.0)*2.0+qrk*tyh*(qrk*qzk-qxk*qyk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*tzh*(qrk*qyk+qxk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*4.0-qrk*txh*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk)*1.0/pow(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk,2.0)*2.0);
            float j12 = w * -(Lx*(qrk*qrk+qxk*qxk-qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Ly*(qrk*qzk*2.0+qxk*qyk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lz*(qrk*qyk*2.0-qxk*qzk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
            float j13 = w * -(Ly*(qrk*qrk-qxk*qxk+qyk*qyk-qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Lx*(qrk*qzk*2.0-qxk*qyk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lz*(qrk*qxk*2.0+qyk*qzk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
            float j14 = w * -(Lz*(qrk*qrk-qxk*qxk-qyk*qyk+qzk*qzk))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)-(Lx*(qrk*qyk*2.0+qxk*qzk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk)+(Ly*(qrk*qxk*2.0-qyk*qzk*2.0))/(qrk*qrk+qxk*qxk+qyk*qyk+qzk*qzk);
#pragma omp critical
{
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_h, j1);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_h + 1, j2);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_h + 2, j3);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_h + 3, j4);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_h + 4, j5);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_h + 5, j6);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_h + 6, j7);

            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_k, j8);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_k + 1, j9);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_k + 2, j10);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_k + 3, j11);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_k + 4, j12);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_k + 5, j13);
            splm_stm_nonzeroval(&jac_st, rows[i] + j, col_k + 6, j14);
}

          }
        }

        {
          pcl::ScopeTime t("splm_stm2ccsm");
          splm_stm2ccsm(&jac_st, jac);
          splm_stm_free(&jac_st);
        }
        //std::cout << "non zeros:" << Jnnz << " " << jac->nnz << std::endl;
      }
    }

    template <typename PointT>
    void MVNonLinearICP<PointT>::compute()
    {
      /*{
        pcl::visualization::PCLVisualizer vis("initial clouds");
        for(size_t i=0; i < input_clouds_.size(); i++)
        {

          std::stringstream cloud_name;
          cloud_name << "cloud_" << i;
          pcl::visualization::PointCloudColorHandlerRandom<PointT> rand_h(input_clouds_[i]);
          vis.addPointCloud<PointT>(input_clouds_[i], rand_h, cloud_name.str());
        }
        vis.spin();
      }*/

      //compute distance transforms for all views...
      dist_transforms_.resize(input_clouds_.size());
      octrees_.resize(input_clouds_.size());

      for(size_t i=0; i < input_clouds_.size(); i++)
      {
        PointTCloudConstPtr target_icp_const(new pcl::PointCloud<PointT>(*input_clouds_[i]));
        /*dist_transforms_[i].reset(new distance_field::PropagationDistanceField<PointT>(dt_vx_size_));
        dist_transforms_[i]->setDistanceExtend(std::max(max_correspondence_distance_, 0.02f));
        dist_transforms_[i]->setHuberSigma(inlier_threshold_ * 1.345);
        dist_transforms_[i]->setInputCloud(target_icp_const);
        dist_transforms_[i]->compute();
        dist_transforms_[i]->computeFiniteDifferences();*/

        octrees_[i].reset (new pcl::octree::OctreePointCloudSearch<PointT> (dt_vx_size_));
        octrees_[i]->setInputCloud (input_clouds_[i]);
        octrees_[i]->addPointsFromInputCloud ();
      }

      if(normals_available_)
      {
          for(size_t i=0; i < input_clouds_.size(); i++)
          {
              assert(input_clouds_[i]->points.size() == input_normals_[i]->points.size());
          }
      }

      //create S based on adjacency matrix
      int Jnnz = 0;
      for(size_t i=0; i < input_clouds_.size(); i++)
      {
        for(size_t j=(i+1); j < input_clouds_.size(); j++)
        {
          if(adjacency_matrix_[i][j])
          {
            std::pair<int, int> p = std::make_pair<int, int>((int)i,(int)j);
            S_.push_back(p);
          }
        }
      }

      int m = 7 * input_clouds_.size();
      int n= 0;
      std::cout << "size of S:" << S_.size() << std::endl;
      for(size_t i =0; i < S_.size(); i++)
      {
        n += input_clouds_[S_[i].first]->points.size();
        Jnnz += input_clouds_[S_[i].first]->points.size() * 14;
      }

      std::cout << m << " " << n << " " << " " << n * m << " " << Jnnz << std::endl;
      double * p;
      p = new double[m];
      double opts[LM_OPTS_SZ], info[LM_INFO_SZ];

      for(size_t i=0; i < input_clouds_.size(); i++)
      {
        p[i*7 + 0]= 0.0;
        p[i*7 + 1]= 0.0;
        p[i*7 + 2]= 0.0;
        p[i*7 + 3]= 1.0;
        p[i*7 + 4]=0.0;
        p[i*7 + 5]=0.0;
        p[i*7 + 6]=0.0;
      }

      //m -= 7;

      opts[0]=LM_INIT_MU; opts[1]=1e-9; opts[2]=1e-9; opts[3]=1e-9;
      opts[4]=LM_DIFF_DELTA; // relevant only if the finite difference Jacobian version is used
      //int ret=dlevmar_dif(MVNLICP::model_func, p, NULL, m, n, 500, opts, info, NULL, NULL, this); // without analytic Jacobian
      if(sparse_solver_)
      {
        pcl::ScopeTime t("sparse solver");
        int ret_crs=sparselm_derccs(MVNLICP::model_func<PointT>, MVNLICP::jac_func_crs<PointT>, p, NULL, m, 0, n, Jnnz ,-1, max_iterations_, NULL, info, this); // with analytic Jacobian
      }
      else
      {
        pcl::ScopeTime t("regular solver");
        int ret=dlevmar_der(MVNLICP::model_func<PointT>, MVNLICP::jac_func<PointT>, p, NULL, m, n, max_iterations_, NULL, info, NULL, NULL, this); // with analytic Jacobian
      }

      printf("Levenberg-Marquardt returned in %g iter, reason %g, sumsq %g [%g]\n", info[5], info[6], info[1], info[3]);

      //pcl::visualization::PCLVisualizer vis("test");
      PointTCloudPtr big_cloud(new pcl::PointCloud<PointT>);
      PointTCloudPtr big_cloud_initial(new pcl::PointCloud<PointT>);

      transformations_.clear();
      for(size_t i=0; i < input_clouds_.size(); i++)
      {

        PointTCloudPtr input_transformed(new pcl::PointCloud<PointT>);

        Eigen::Quaterniond rot(p[i * 7 + 3], p[i * 7], p[i * 7 + 1], p[i * 7 + 2]);
        Eigen::Vector3d trans(p[i * 7 + 4], p[i * 7 + 5], p[i * 7 + 6]);
        Eigen::Matrix3d R = MVNLICP::coolquat_to_mat(rot);
        R /= rot.squaredNorm();


        if(MVNLICP_WEIRD_TRANS_)
        {
          input_transformed->points.resize (input_clouds_[i]->points.size ());
          for (size_t k = 0; k < input_clouds_[i]->points.size (); k++)
          {
            input_transformed->points[k].getVector3fMap () = R.cast<float> () * (input_clouds_[i]->points[k].getVector3fMap ()
                + trans.cast<float> ());
          }
        }
        else
        {
          Eigen::Matrix4f T_h;
          T_h.setIdentity();
          T_h.block<3,3>(0,0) = R.cast<float>();
          T_h.block<3,1>(0,3) = trans.cast<float>();
          pcl::transformPointCloud(*input_clouds_[i], *input_transformed, T_h);
          transformations_.push_back(T_h);
        }


        std::stringstream cloud_name;
        cloud_name << "cloud_" << i;
        //pcl::visualization::PointCloudColorHandlerRandom<PointT> rand_h(input_transformed);
        //vis.addPointCloud<PointT>(input_transformed, rand_h, cloud_name.str());

        /*pcl::visualization::PointCloudColorHandlerRandom<PointT> rand_h(input_clouds_[i]);
        vis.addPointCloud<PointT>(input_clouds_[i], rand_h, cloud_name.str());*/

        printf("Best fit parameters: %.7g %.7g %.7g %.7g %.7g %.7g %.7g\n", p[i * 7 +0], p[i * 7 +1], p[i * 7 +2],
                                                                             p[i * 7 +3], p[i * 7 +4], p[i * 7 +5], p[i * 7 + 6]);

        *big_cloud += *input_transformed;
        *big_cloud_initial += *input_clouds_[i];
      }

      //vis.addCoordinateSystem(0.1f);
      //vis.spin();

      /*{
        pcl::visualization::PCLVisualizer vis("big clouds");
        int v1, v2;
        vis.createViewPort(0,0,0.5,1,v1);
        vis.createViewPort(0.5,0,1,1,v2);
        float leaf=0.001f;
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setMeanK (50);
        sor.setStddevMulThresh (2);

        {
//          PointTCloudPtr filtered(new pcl::PointCloud<PointT>);
//          sor.setInputCloud (big_cloud);
//          sor.filter (*filtered);
//
//          {
//            pcl::VoxelGrid<PointT> sor;
//            sor.setLeafSize (leaf,leaf,leaf);
//            sor.setInputCloud (filtered);
//            sor.filter (*big_cloud);
//          }

          pcl::visualization::PointCloudColorHandlerCustom<PointT> rand_h(big_cloud, 255, 0, 0);
          vis.addPointCloud<PointT>(big_cloud, rand_h, "aligned", v2);
        }

        {
//          PointTCloudPtr filtered(new pcl::PointCloud<PointT>);
//          sor.setInputCloud (big_cloud_initial);
//          sor.filter (*filtered);
//
//          {
//            pcl::VoxelGrid<PointT> sor;
//            sor.setInputCloud (filtered);
//            sor.setLeafSize (leaf,leaf,leaf);
//            sor.filter (*big_cloud_initial);
//          }

          pcl::visualization::PointCloudColorHandlerCustom<PointT> rand_h(big_cloud_initial, 0, 255, 0);
          vis.addPointCloud<PointT>(big_cloud_initial, rand_h, "initial", v1);
        }

        vis.spin();
        vis.removeAllPointClouds();

        {
          pcl::visualization::PointCloudColorHandlerCustom<PointT> rand_h(big_cloud, 255, 0, 0);
          vis.addPointCloud<PointT>(big_cloud, rand_h, "aligned", v2);
        }

        {
          pcl::visualization::PointCloudColorHandlerCustom<PointT> rand_h(big_cloud_initial, 0, 255, 0);
          vis.addPointCloud<PointT>(big_cloud_initial, rand_h, "initial", v2);
        }

        vis.spin();
      }*/

      /*{

        PointTCloudPtr filtered(new pcl::PointCloud<PointT>);
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud (big_cloud);
        sor.setMeanK (50);
        sor.setStddevMulThresh (0.5);
        sor.filter (*filtered);

        pcl::visualization::PCLVisualizer vis("big cloud (sor)");
        pcl::visualization::PointCloudColorHandlerCustom<PointT> rand_h(filtered, 255, 0, 0);
        vis.addPointCloud<PointT>(filtered);
        vis.spin();
      }*/

      delete[] p;
      return;
    }
  }
}

template class PCL_EXPORTS faat_pcl::registration::MVNonLinearICP<pcl::PointXYZ>;
template class PCL_EXPORTS faat_pcl::registration::MVNonLinearICP<pcl::PointXYZRGB>;
