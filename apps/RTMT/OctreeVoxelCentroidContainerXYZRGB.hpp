
#ifndef V4R_OCTREE_VOXELCENTROID_CONTAINER_HPP
#define V4R_OCTREE_VOXELCENTROID_CONTAINER_HPP

#include <pcl/common/point_operators.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/octree/octree_pointcloud.h>

namespace pcl
{
  namespace octree
  {
    /** \brief @b Octree pointcloud voxel centroid leaf node class
      * \note This class implements a leaf node that calculates the mean centroid of PointXYZRGB points added this octree container.
      */
    template<typename PointT=pcl::PointXYZRGB>
    class OctreeVoxelCentroidContainerXYZRGB : public OctreeContainerBase
    {
      public:
        /** \brief Class initialization. */
        OctreeVoxelCentroidContainerXYZRGB ()
        {
          this->reset();
        }

        /** \brief Empty class deconstructor. */
        virtual ~OctreeVoxelCentroidContainerXYZRGB ()
        {
        }

        /** \brief deep copy function */
        virtual OctreeVoxelCentroidContainerXYZRGB *
        deepCopy () const
        {
          return (new OctreeVoxelCentroidContainerXYZRGB (*this));
        }

        /** \brief Equal comparison operator - set to false
         */
         // param[in] OctreeVoxelCentroidContainerXYZRGB to compare with
        virtual bool operator==(const OctreeContainerBase&) const
        {
          return ( false );
        }

        /** \brief Add new point to voxel.
          * \param[in] new_point the new point to add  
          */
        void 
        addPoint (const PointT& new_point)
        {
          using namespace pcl::common;

          ++point_counter_;

          pt_[0] += double(new_point.x);
          pt_[1] += double(new_point.y);
          pt_[2] += double(new_point.z);
          r_ += unsigned(new_point.r);
          g_ += unsigned(new_point.g);
          b_ += unsigned(new_point.b);
        }

        /** \brief Calculate centroid of voxel.
          * \param[out] centroid_arg the resultant centroid of the voxel 
          */
        void 
        getCentroid (PointT& centroid_arg) const
        {
          using namespace pcl::common;

          if (point_counter_)
          {
            centroid_arg.getVector3fMap() = (pt_ / static_cast<double> (point_counter_)).template cast<float>();
            centroid_arg.r = static_cast<unsigned char>(r_ / point_counter_);
            centroid_arg.g = static_cast<unsigned char>(g_ / point_counter_);
            centroid_arg.b = static_cast<unsigned char>(b_ / point_counter_);
          }
          else
          {
            centroid_arg.x = std::numeric_limits<float>::quiet_NaN();
            centroid_arg.y = std::numeric_limits<float>::quiet_NaN();
            centroid_arg.z = std::numeric_limits<float>::quiet_NaN();
            centroid_arg.r = 0;
            centroid_arg.g = 0;
            centroid_arg.b = 0;
          }
        }

        /** \brief Reset leaf container. */
        virtual void 
        reset ()
        {
          using namespace pcl::common;

          point_counter_ = 0;
          pt_ = Eigen::Vector3d(0.,0.,0.);
          r_ = g_ = b_ = 0;
        }

      private:
        unsigned int point_counter_;
        Eigen::Vector3d pt_;
        unsigned r_, g_, b_;
    };

  }
}

#endif

