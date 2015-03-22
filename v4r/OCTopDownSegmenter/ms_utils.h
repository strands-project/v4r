
#ifndef V4R_MUMFORD_SHAH_UTILS
#define V4R_MUMFORD_SHAH_UTILS


#undef NDEBUG
#include <assert.h>

#include <pcl/common/common.h>
#include <pcl/common/time.h>
#include <pcl/common/eigen.h>
#include "v4rexternal/opennurbs/opennurbs.h"
#include <pcl/common/pca.h>
#include <set>

namespace v4rOCTopDownSegmenter
{
    static int PLANAR_MODEL_TYPE_ = 0;
    static int BSPLINE_MODEL_TYPE_3x3 = 1;
    static int BSPLINE_MODEL_TYPE_5x5 = 2;

    class StopWatch
    {
      public:
        /** \brief Constructor. */
        StopWatch () : start_time_ (boost::posix_time::microsec_clock::local_time ())
        {
        }

        /** \brief Destructor. */
        virtual ~StopWatch () {}

        /** \brief Retrieve the time in milliseconds spent since the last call to \a reset(). */
        inline double
        getTime ()
        {
          boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
          return (static_cast<double> (((end_time - start_time_).total_milliseconds ())));
        }

        inline double
        getTimeMicroSeconds ()
        {
          boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
          return (static_cast<double> (((end_time - start_time_).total_microseconds ())));
        }

        /** \brief Retrieve the time in seconds spent since the last call to \a reset(). */
        inline double
        getTimeSeconds ()
        {
          return (getTime () * 0.001f);
        }

        /** \brief Reset the stopwatch to 0. */
        inline void
        reset ()
        {
          start_time_ = boost::posix_time::microsec_clock::local_time ();
        }

      protected:
        boost::posix_time::ptime start_time_;
    };

    //USEFUL CLASSES AND HELPER FUNCTIONS

    class Line
    {
    public:

        Line()
        {

        }

        Line(Eigen::Vector3f & origin, Eigen::Vector3f & direction)
        {
            origin_ = origin;
            direction_ = direction;
            direction_.normalize();
        }

        Eigen::Vector3f eval(float t)
        {
            return origin_ + t * direction_;
        }

        float evalZ(float t)
        {
            return origin_[2] + t * direction_[2];
        }

        Eigen::Vector3f origin_;
        Eigen::Vector3f direction_;
    };

    inline Eigen::Vector3f intersectLineWithPlane(Line & l, Eigen::Vector3f & n, float d, Eigen::Vector3f & pp)
    {
        float los_distance = (pp - l.origin_).dot(n) / (l.direction_.dot(n));
        return l.eval(los_distance);
    }

    inline float intersectLineWithPlaneZValue(Line & l, Eigen::Vector3f & n, float d, Eigen::Vector3f & pp)
    {
        return l.evalZ((pp - l.origin_).dot(n) / (l.direction_.dot(n)));
    }

    inline float lineToPlaneDistance(Line & l, Eigen::Vector3f & n, float d, Eigen::Vector3f & pp)
    {
        return (pp - l.origin_).dot(n) / (l.direction_.dot(n));
    }

    //END USEFUL CLASSES

    template<typename PointT>
    class Region
    {
        public:
            Region()
            {
                valid_ = true;
                plane_model_defined_ = false;
                mean_ = Eigen::Vector3d(-1,-1,-1);
                for(size_t i=0; i < 9; i++)
                {
                    accu_[i] = 0.0;
                }

                color_mean_ = Eigen::Vector3d(0,0,0);

                current_model_type_ = PLANAR_MODEL_TYPE_;
                smoothness_ = 0.f;
            }

            ~Region()
            {

            }

            float computePlaneErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                         std::vector<int> & indices,
                                         Eigen::Vector3f & nn, float d, Eigen::Vector3f & point_on_plane,
                                         std::vector<Line> & los_);

            float computeColorErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                         std::vector<Eigen::Vector3d> & lab_values,
                                         std::vector<int> & indices);

            float computeColorError(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                    std::vector<Eigen::Vector3d> & lab_values);

            float computePlaneError(typename pcl::PointCloud<PointT>::Ptr & cloud, std::vector<Line> & los_);

            float computePointToPlaneError(typename pcl::PointCloud<PointT>::Ptr & cloud);

            float getModelTypeError()
            {
                if(current_model_type_ == PLANAR_MODEL_TYPE_)
                {
                    return planar_error_;
                }
                else if(current_model_type_ == BSPLINE_MODEL_TYPE_3x3 || current_model_type_ == BSPLINE_MODEL_TYPE_5x5)
                {
                    return bspline_error_;
                }
            }

            float mComp()
            {

                return smoothness_; //0 for planes, something else for B-splines

                /*if(current_model_type_ == PLANAR_MODEL_TYPE_)
                {
                    return 0.f;
                }
                else if(current_model_type_ == BSPLINE_MODEL_TYPE_3x3)
                {
                    return 1.f;
                }*/
            }

            /*float getCurvatureSqr(typename pcl::PointCloud<PointT>::Ptr & cloud)
            {
                if(current_model_type_ == PLANAR_MODEL_TYPE_)
                {
                    return 0.f;
                }
                else if(current_model_type_ == BSPLINE_MODEL_TYPE_3x3 || current_model_type_ == BSPLINE_MODEL_TYPE_5x5)
                {
                    double curvature_sqr(0.0);
                    double ptd[6];
                    for(size_t i=0; i<indices_.size(); i++)
                    {
                        int u = i % cloud->width;
                        int v = i / cloud->width;
                        bspline_model_.Evaluate(u,v,2,1,ptd);
                        double c = (ptd[3]+ptd[5])*0.5;
                        curvature_sqr += (c*c);
                    }
                    curvature_sqr /= indices_.size();
                    return static_cast<float>(curvature_sqr);
                }
            }*/

            float getColorErrorForPoint(std::vector<Eigen::Vector3d> & lab_values,
                                        int idx)
            {
                return (lab_values[idx] - color_mean_).squaredNorm();
            }

            float getModelErrorForPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                        int idx,
                                        std::vector<Line> & los_)
            {
                if(current_model_type_ == PLANAR_MODEL_TYPE_)
                {
                    if(plane_model_defined_)
                    {
                        Eigen::Vector3f point_on_plane = mean_.cast<float>();

                        Eigen::Vector3f p = cloud->at(idx).getVector3fMap();
                        Eigen::Vector3f v = p - point_on_plane;
                        Eigen::Vector3f proj = p - v.dot( planar_model_.first) *  planar_model_.first;
                        return (proj - p).norm();

                        /*Eigen::Vector3f intersect = intersectLineWithPlane(los_[idx], planar_model_.first,
                                                                                      planar_model_.second, point_on_plane);*/

                        //return (intersect - cloud->points[idx].getVector3fMap()).squaredNorm();

                        /*float z_intersect = intersectLineWithPlaneZValue(los_[idx], planar_model_.first,
                                                                         planar_model_.second, point_on_plane);

                        return (cloud->points[idx].z - z_intersect) * (cloud->points[idx].z - z_intersect);*/
                    }
                    else
                    {
                        //maybe nothing was fitted to this point (supervoxel region)
                        //i should not return 0.f
                        return 0.0001f;
                    }
                }
                else if(current_model_type_ == BSPLINE_MODEL_TYPE_3x3 || current_model_type_ == BSPLINE_MODEL_TYPE_5x5)
                {
                    PointT proj;
                    pca_bspline_->project(cloud->points[idx], proj);

                    double P[3];
                    bspline_model_.Evaluate (proj.x, proj.y, 0, bspline_model_.Dimension(), P);
                    PointT p1, p2;
                    p1.x = P[0];    p1.y = P[1];    p1.z = P[2];
                    pca_bspline_->reconstruct(p1, p2);

                    return (cloud->points[idx].getVector3fMap() - p2.getVector3fMap()).norm();

                    /*double u,v,z;
                    u = idx % cloud->width;
                    v = idx / cloud->width;

                    bspline_model_.Evaluate(u, v, 0, 1, &z);

                    Eigen::Vector3f p = cloud->points[idx].getVector3fMap();
                    return (z - p.norm()) * (z - p.norm());*/
                }

                return 0.f;

            }

            float getModelErrorForPointUnorganized(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                        int idx)
            {
                if(current_model_type_ == PLANAR_MODEL_TYPE_)
                {
                    if(plane_model_defined_)
                    {
                        Eigen::Vector3f point_on_plane = mean_.cast<float>();
                        Eigen::Vector3f p = cloud->at(idx).getVector3fMap();
                        Eigen::Vector3f v = p - point_on_plane;
                        return std::abs(v.dot(planar_model_.first));
                    }
                    else
                    {
                        //maybe nothing was fitted to this point (supervoxel region)
                        //i should not return 0.f
                        return 0.0001f;
                    }
                }
                else if(current_model_type_ == BSPLINE_MODEL_TYPE_3x3 || current_model_type_ == BSPLINE_MODEL_TYPE_5x5)
                {

                    double u,v,z;
                    u = idx % cloud->width;
                    v = idx / cloud->width;

                    /*ON_NurbsSurface& bspline = regions[i]->bspline_model_;
                    if(u<bspline.Knot(0,0) || u>bspline.Knot(0,bspline.KnotCount(0)-1))
                      throw std::runtime_error("[v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter::visualizeSegmentation] Error, index u out of bounds.");
                    if(v<bspline.Knot(1,0) || v>bspline.Knot(1,bspline.KnotCount(1)-1))
                      throw std::runtime_error("[v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter::visualizeSegmentation] Error, index v out of bounds.");*/

                    bspline_model_.Evaluate(u, v, 0, 1, &z);

                    Eigen::Vector3f p = cloud->points[idx].getVector3fMap();
                    return (z - p.norm()) * (z - p.norm());
                    //cloud_cc->at(regions[i]->indices_[j]).getVector3fMap() = l.eval(z);

                    //PCL_WARN("Implement this...\n");
                }

                return 0.f;

            }

            int id_;
            bool valid_; //once two regions gets merged, one of them becomes invalid and the other contains a merge of both
            std::vector<int> indices_; //indices to original cloud
            std::vector<int> indices_downsampled_; //indices to original cloud (downsampled)
            std::set<int> indices_set_; //hack to remove and add indices fast
            Eigen::Vector3d mean_;
            EIGEN_ALIGN16 Eigen::Matrix3d covariance_;
            Eigen::Matrix<double, 1, 9, Eigen::RowMajor> accu_;

            float planar_error_;
            bool plane_model_defined_;
            std::pair<Eigen::Vector3f, float> planar_model_;

            float bspline_error_;
            ON_NurbsSurface bspline_model_;
            boost::shared_ptr<pcl::PCA<PointT> > pca_bspline_;
            bool bspline_model_defined_;
            int current_model_type_;
            float smoothness_; //sum of curvature

            //color model
            Eigen::Vector3d color_mean_;
            float color_error_;
    };

    template<typename PointT>
    class MergeCandidate
    {
        public:

            MergeCandidate()
            {
                valid_ = false;
                plane_model_defined_ = false;
                boundary_length_ = 0;
                current_model_type_ = PLANAR_MODEL_TYPE_;
                smoothness_ = 0.f;
                color_mean_ = Eigen::Vector3d(0,0,0);

            }

            ~MergeCandidate()
            {

            }

            float computePlaneErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                         std::vector<int> & indices,
                                         Eigen::Vector3f & nn, float d, Eigen::Vector3f & point_on_plane,
                                         std::vector<Line> & los_);

            float computePointToPlaneErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                         std::vector<int> & indices,
                                         Eigen::Vector3f & nn, float d, Eigen::Vector3f & point_on_plane);

            float computeColorErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                         std::vector<Eigen::Vector3d> & lab_values,
                                         std::vector<int> & indices);

            float computeColorError(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                    std::vector<Eigen::Vector3d> & lab_values);

            float computePlaneError(typename pcl::PointCloud<PointT>::Ptr & cloud, std::vector<Line> & los_);

            float computePointToPlaneError(typename pcl::PointCloud<PointT>::Ptr & cloud);

            float computeBsplineError(const Eigen::VectorXd& cloud_z, int cloud_width,
                                      int order=3, int CPx=3, int CPy=3);

            float computeBsplineErrorUnorganized(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                                 int order=3, int CPx=3, int CPy=3);

            void merge(std::vector< std::vector<int> > & adjacent,
                       std::vector<MergeCandidate<PointT> > & candidates,
                       std::vector< std::pair<int, int> > & recompute_candidates);

            bool isValid()
            {
                return valid_ && r1_->valid_ && r2_->valid_;
            }

            void increaseModelType()
            {
                current_model_type_++;
            }

            int getModelType()
            {
               return current_model_type_;
            }

            float getModelTypeError()
            {
                if(current_model_type_ == PLANAR_MODEL_TYPE_)
                {
                    return planar_error_;
                }
                else if(current_model_type_ == BSPLINE_MODEL_TYPE_3x3 || current_model_type_ == BSPLINE_MODEL_TYPE_5x5)
                {
                    return bspline_error_;
                }
            }

            float mComp()
            {

                return smoothness_;

                /*if(current_model_type_ == PLANAR_MODEL_TYPE_)
                {
                    return 0.f;
                }
                else if(current_model_type_ == BSPLINE_MODEL_TYPE_3x3)
                {
                    return 1.f;
                }*/

            }

            bool valid_;
            boost::shared_ptr<Region<PointT> > r1_, r2_;
            float min_nyu_;
            float planar_error_;
            float bspline_error_;
            float color_error_;

            Eigen::Vector3d mean_;
            EIGEN_ALIGN16 Eigen::Matrix3d covariance_;
            Eigen::Matrix<double, 1, 9, Eigen::RowMajor> accu_;
            int boundary_length_;

            //color model
            Eigen::Vector3d color_mean_;

            bool plane_model_defined_;
            std::pair<Eigen::Vector3f, float> planar_model_;

            ON_NurbsSurface bspline_model_;
            boost::shared_ptr<pcl::PCA<PointT> > pca_bspline_;
            bool bspline_model_defined_;

            int current_model_type_;

            float smoothness_;
    };

    class PixelMove
    {
        public:
            int current_region_;
            int r1_;
            int r2_;
            int boundary_length_delta_;
            float data_error_delta_;
            float color_error_delta_;
            bool valid_;
            bool recompute_;
            int idx_;
            float improvement_;
    };

    struct pixelMoveComp {
        bool operator() (const PixelMove & lhs, const PixelMove & rhs) const
        {
            return lhs.improvement_ < rhs.improvement_;
        }
    };
}

#endif
