/*
 * lm_icp.h
 *
 *  Created on: Jun 12, 2013
 *      Author: aitor
 */

#include "faat_pcl/registration/lm_icp.h"
//#include "pcl/visualization/pcl_visualizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

namespace faat_pcl
{
    namespace registration
    {
        //pcl::visualization::PCLVisualizer vis_("lm_icp");
        bool angle_axis = false;
        double last_error = std::numeric_limits<double>::infinity();

        inline double huber(double d, double sigma)
        {
            if(d <= sigma)
            {
                return (d*d) / 2;
            }
            else
            {
                return sigma * (d - sigma / 2); //k * d - ((d*d) / 2);
                //return sigma;
            }
        }

        inline double huberFitzigibon(double d, double sigma)
        {
            if(d <= sigma)
            {
                return (d*d);
            }
            else
            {
                return 2 * sigma * d - sigma * sigma;
                //return sigma;
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

        void model_func(double *p, double *x, int m, int n, void *data)
        {
            NonLinearICP * nl_icp = (NonLinearICP *)(data);
            //double inliers_threshold_ = nl_icp->dt_vx_size_;
            double huber_sigma = nl_icp->inlier_threshold_ * 1.345;

            for(size_t k=0; k < n; k++)
            {
                //x[k] = huberFitzigibon(huber_sigma*2.f, huber_sigma);
                x[k] = 0;
            }

            Eigen::Quaterniond rot(p[3],p[0],p[1],p[2]);
            Eigen::Vector3d trans(p[4],p[5],p[6]);
            Eigen::Matrix3d R = coolquat_to_mat(rot);
            R /= rot.squaredNorm();
            Eigen::Matrix4f T;
            T.setIdentity();
            T.block<3,3>(0,0) = R.cast<float>();
            T.block<3,1>(0,3) = trans.cast<float>();

            nl_icp->input_transformed_.reset(new pcl::PointCloud<NonLinearICP::PointT>);
            pcl::transformPointCloud(*nl_icp->input_, *nl_icp->input_transformed_, T);

            //find correspondences from nl_icp->input_ (transformed) to nl_icp->target_
            //and update measurements vector
            int n_matches = 0;
            int idx_match;
            float distance;
            float color_distance = -1.f;

            std::vector<double> distances(nl_icp->input_->points.size(), std::numeric_limits<double>::infinity());

            for(size_t k=0; k < n; k++)
            {

                idx_match = -1;
                if(nl_icp->use_octree_)
                {
                    std::vector<int> pointIdxNKNSearch;
                    std::vector<float> pointNKNSquaredDistance;
                    if (nl_icp->octree_->nearestKSearch (nl_icp->input_transformed_->points[k], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        idx_match = pointIdxNKNSearch[0];
                        distance = sqrt(pointNKNSquaredDistance[0]);
                    }
                }
                else
                {
                    nl_icp->dist_trans_->getCorrespondence(nl_icp->input_transformed_->points[k], &idx_match, &distance, -1.f, &color_distance);
                }

                distances[k] = distance;
            }

            for(size_t j=0; j < nl_icp->input_->points.size(); j++)
            {
                n_matches++;
                x[j] = huberFitzigibon(distances[j], huber_sigma);
            }

            double sum=0;
            for(size_t k=0; k < n; k++)
            {
                sum += x[k] * x[k];
            }

            std::cout << "residuals sum:" << sum << " " << n_matches << std::endl;
        }

        void jac_func(double *p, double *jac, int m, int n, void *data)
        {
            register int k, j;
            NonLinearICP * nl_icp = (NonLinearICP *)(data);

            double huber_sigma = nl_icp->inlier_threshold_ * 1.345;

            Eigen::Matrix4d transform;
            transform.setIdentity();

            Eigen::Quaterniond rot(p[3],p[0],p[1],p[2]);
            Eigen::Vector3d trans(p[4],p[5],p[6]);
            Eigen::Matrix3d R = coolquat_to_mat(rot);
            R /= rot.squaredNorm();
            transform.block<3,3>(0,0) = R;
            transform.block<3,1>(0,3) = trans;

            nl_icp->input_transformed_.reset(new pcl::PointCloud<NonLinearICP::PointT>);
            pcl::transformPointCloud(*nl_icp->input_, *nl_icp->input_transformed_, transform);

            float x,y,z;
            float Lx,Ly,Lz;
            float qx,qy,qz,qr,tx,ty,tz;
            qx = p[0];
            qy = p[1];
            qz = p[2];
            qr = p[3];
            tx = p[4];
            ty = p[5];
            tz = p[6];

            /* fill Jacobian row by row */
            int used_for_jac = 0;
            for(k=j=0; k<n; ++k)
            {
                if(!nl_icp->use_octree_)
                {
                    Eigen::Vector3f L;
                    nl_icp->dist_trans_->getDerivatives(nl_icp->input_transformed_->points[k], L);
                    Lx = L[0];
                    Ly = L[1];
                    Lz = L[2];
                }
                else
                {
                    float step = 0.002f;

                    std::vector<int> pointIdxNKNSearch;
                    std::vector<float> pointNKNSquaredDistance;

                    pcl::PointXYZ p_plus, p_minus;
                    int idx_match_plus, idx_match_minus;
                    float distance_plus, distance_minus;
                    pcl::PointXYZ p = nl_icp->input_transformed_->points[k];
                    { //X
                        p_plus = p;
                        p_minus = p;
                        p_plus.x += step / 2.f;
                        p_minus.x -= step / 2.f;

                        if (nl_icp->octree_->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                        {
                            idx_match_plus = pointIdxNKNSearch[0];
                            distance_plus = sqrt(pointNKNSquaredDistance[0]);
                        }

                        if (nl_icp->octree_->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
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

                        if (nl_icp->octree_->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                        {
                            idx_match_plus = pointIdxNKNSearch[0];
                            distance_plus = sqrt(pointNKNSquaredDistance[0]);
                        }

                        if (nl_icp->octree_->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
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

                        if (nl_icp->octree_->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                        {
                            idx_match_plus = pointIdxNKNSearch[0];
                            distance_plus = sqrt(pointNKNSquaredDistance[0]);
                        }

                        if (nl_icp->octree_->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                        {
                            idx_match_minus = pointIdxNKNSearch[0];
                            distance_minus = sqrt(pointNKNSquaredDistance[0]);
                        }

                        Lz = sqrt(huberFitzigibon(distance_plus, huber_sigma)) - sqrt(huberFitzigibon(distance_minus, huber_sigma));
                    }
                }

                if(pcl_isnan(Lx) || pcl_isnan(Ly) || pcl_isnan(Lz))
                {
                    //PCL_WARN("Derivatives are NaN...\n");
                    jac[j++]=0;
                    jac[j++]=0;
                    jac[j++]=0;
                    jac[j++]=0;
                    jac[j++]=0;
                    jac[j++]=0;
                    jac[j++]=0;
                    continue;
                }

                x = nl_icp->input_transformed_->points[k].x;
                y = nl_icp->input_transformed_->points[k].y;
                z = nl_icp->input_transformed_->points[k].z;

                jac[j++] = Ly*((qy*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qx*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qr*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+qx*x*(qr*qz*2.0-qx*qy*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qx*z*(qr*qx*2.0+qy*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qx*y*(qr*qr-qx*qx+qy*qy-qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0)-Lz*((qz*x*-2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qr*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qx*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+qx*x*(qr*qy*2.0+qx*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qx*y*(qr*qx*2.0-qy*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qx*z*(qr*qr-qx*qx-qy*qy+qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0)+Lx*((qx*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qy*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qz*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-qx*y*(qr*qz*2.0+qx*qy*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qx*z*(qr*qy*2.0-qx*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qx*x*(qr*qr+qx*qx-qy*qy-qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0);
                jac[j++] = -Lx*((qy*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qx*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qr*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+qy*y*(qr*qz*2.0+qx*qy*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qy*z*(qr*qy*2.0-qx*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qy*x*(qr*qr+qx*qx-qy*qy-qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0)+Lz*((qr*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qz*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qy*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-qy*x*(qr*qy*2.0+qx*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qy*y*(qr*qx*2.0-qy*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qy*z*(qr*qr-qx*qx-qy*qy+qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0)+Ly*((qx*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qy*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qz*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+qy*x*(qr*qz*2.0-qx*qy*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qy*z*(qr*qx*2.0+qy*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qy*y*(qr*qr-qx*qx+qy*qy-qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0);
                jac[j++] = -Lx*((qz*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qr*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qx*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+qz*y*(qr*qz*2.0+qx*qy*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qz*z*(qr*qy*2.0-qx*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qz*x*(qr*qr+qx*qx-qy*qy-qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0)-Ly*((qr*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qz*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qy*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-qz*x*(qr*qz*2.0-qx*qy*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qz*z*(qr*qx*2.0+qy*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qz*y*(qr*qr-qx*qx+qy*qy-qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0)+Lz*((qx*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qy*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qz*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-qz*x*(qr*qy*2.0+qx*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qz*y*(qr*qx*2.0-qy*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qz*z*(qr*qr-qx*qx-qy*qy+qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0);
                jac[j++] = Lx*((qr*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qz*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qy*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-qr*y*(qr*qz*2.0+qx*qy*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qr*z*(qr*qy*2.0-qx*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qr*x*(qr*qr+qx*qx-qy*qy-qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0)-Ly*((qz*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qr*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qx*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-qr*x*(qr*qz*2.0-qx*qy*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qr*z*(qr*qx*2.0+qy*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qr*y*(qr*qr-qx*qx+qy*qy-qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0)+Lz*((qy*x*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-(qx*y*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)+(qr*z*2.0)/(qr*qr+qx*qx+qy*qy+qz*qz)-qr*x*(qr*qy*2.0+qx*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0+qr*y*(qr*qx*2.0-qy*qz*2.0)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0-qr*z*(qr*qr-qx*qx-qy*qy+qz*qz)*1.0/pow(qr*qr+qx*qx+qy*qy+qz*qz,2.0)*2.0);
                jac[j++] = Lx;
                jac[j++] = Ly;
                jac[j++] = Lz;

                used_for_jac++;
            }

            std::cout << "jacobian: " << j << " " << m*n << " " << used_for_jac << std::endl;
        }

        void NonLinearICP::compute()
        {

            std::cout << input_->points.size() << " " << target_->points.size() << std::endl;
            int m;
            if(angle_axis)
            {
                m = 6;
            }
            else
            {
                m = 7;
            }

            const int n=static_cast<int>(input_->points.size());
            double p[m], x[n], opts[LM_OPTS_SZ], info[LM_INFO_SZ];

            if(angle_axis)
            {
                p[0]=0.0; p[1]=M_PI*2; p[2]=0.0;
                p[3]=0.0; p[4]=0.0; p[5]=0.0;
                printf("Initial parameters: %.7g %.7g %.7g %.7g %.7g %.7g\n", p[0], p[1], p[2], p[3], p[4], p[5]);
            }
            else
            {
                //initialize p (unit quaternion (4) + vector (3))
                Eigen::Quaternionf rotation;
                rotation.setIdentity();
                p[0]=rotation.x(); p[1]=rotation.y(); p[2]=rotation.z(); p[3]=rotation.w();
                p[4]=0.0; p[5]=0.0; p[6]=0.0;
                printf("Initial parameters: %.7g %.7g %.7g %.7g %.7g %.7g %.7g\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
            }

            //distance transform for target
            PointTCloudConstPtr target_icp_const(new pcl::PointCloud<PointT>(*target_));
            dist_trans_.reset(new distance_field::PropagationDistanceField<PointT>(dt_vx_size_));
            //dist_trans_.reset(new faat_pcl::rec_3d_framework::VoxelGridDistanceTransform<PointT>(dt_vx_size_));
            dist_trans_->setDistanceExtend(0.05f);
            dist_trans_->setHuberSigma(dt_vx_size_);
            dist_trans_->setInputCloud(target_icp_const);
            dist_trans_->compute();
            dist_trans_->computeFiniteDifferences();


            octree_.reset (new pcl::octree::OctreePointCloudSearch<PointT> (dt_vx_size_));
            octree_->setInputCloud (target_);
            octree_->addPointsFromInputCloud ();

            //opts[0]=LM_INIT_MU; opts[1]=1E-25; opts[2]=1E-25; opts[3]=1E-25;
            opts[0]=LM_INIT_MU; opts[1]=std::numeric_limits<double>::min(); opts[2]=std::numeric_limits<double>::min(); opts[3]=std::numeric_limits<double>::min();
            opts[4]=LM_DIFF_DELTA; // relevant only if the finite difference Jacobian version is used
            //int ret=dlevmar_dif(model_func, p, NULL, m, n, 500, opts, info, NULL, NULL, this); // without analytic Jacobian
            int ret=dlevmar_der(model_func, jac_func, p, NULL, m, n, iterations_, opts, info, NULL, NULL, this); // with analytic Jacobian

            if(angle_axis)
            {
                printf("Best fit parameters: %.7g %.7g %.7g %.7g %.7g %.7g\n", p[0], p[1], p[2], p[3], p[4], p[5]);
            }
            else
            {
                printf("Best fit parameters: %.7g %.7g %.7g %.7g %.7g %.7g %.7g\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
            }
            printf("Levenberg-Marquardt returned in %g iter, reason %g, sumsq %g [%g]\n", info[5], info[6], info[1], info[3]);


            Eigen::Matrix4d transform;
            transform.setIdentity();

            PointTCloudPtr input_transformed(new pcl::PointCloud<PointT>);

            Eigen::Quaterniond rot(p[3],p[0],p[1],p[2]);
            //rot.normalize();
            Eigen::Vector3d trans(p[4],p[5],p[6]);
            Eigen::Matrix3d R = coolquat_to_mat(rot);
            R /= rot.squaredNorm();
            //transform.block<3,3>(0,0) = rot.toRotationMatrix();
            transform.block<3,3>(0,0) = R;
            //trans = Eigen::Vector3d::Zero();
            transform.block<3,1>(0,3) = trans;
            input_transformed.reset(new pcl::PointCloud<NonLinearICP::PointT>);

            pcl::transformPointCloud(*input_, *input_transformed, transform);

            final_transform_ = transform;
        }
    }
}
