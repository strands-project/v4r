/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
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
 *   * Neither the name of the Willow Garage nor the names of its
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
 *********************************************************************/

/** \author Mrinal Kalakrishnan, Ken Anderson */

#ifndef DF_PROPAGATION_DISTANCE_FIELD_H_
#define DF_PROPAGATION_DISTANCE_FIELD_H_

#include "voxel_grid.h"
#include "distance_field.h"
#include <vector>
#include <list>
#include <eigen3/Eigen/Core>
#include <set>
#include <map>

namespace distance_field
{

  // TODO: Move to voxel_grid.h
  /// \brief Structure the holds the location of voxels withing the voxel map
  typedef Eigen::Vector3i int3;

  // less-than Comparison
  struct compareInt3
  {
    bool
    operator() (int3 loc_1, int3 loc_2) const
    {
      if (loc_1.z () != loc_2.z ())
        return (loc_1.z () < loc_2.z ());
      else if (loc_1.y () != loc_2.y ())
        return (loc_1.y () < loc_2.y ());
      else if (loc_1.x () != loc_2.x ())
        return (loc_1.x () < loc_2.x ());
      return false;
    }
  };

  /**
   * \brief Structure that holds voxel information for the DistanceField.
   */
  struct PropDistanceFieldVoxel
  {
    PropDistanceFieldVoxel ();
    PropDistanceFieldVoxel (int distance_sq);

    int distance_square_; /**< Squared distance from the closest obstacle */
    int3 location_; /**< Grid location of this voxel */
    int3 closest_point_; /**< Closes obstacle from this voxel */
    int update_direction_; /**< Direction from which this voxel was updated */
    bool occupied_;
    int idx_to_input_cloud_; /** Set only if occupied */

    static const int UNINITIALIZED = -1;
  };

  struct SignedPropDistanceFieldVoxel : public PropDistanceFieldVoxel
  {
    SignedPropDistanceFieldVoxel ();
    SignedPropDistanceFieldVoxel (int distance_sq_positive, int distance_sq_negative);
    int positive_distance_square_;
    int negative_distance_square_;
    int3 closest_positive_point_;
    int3 closest_negative_point_;

    static const int UNINITIALIZED = -999;
  };

  /**
   * \brief A DistanceField implementation that uses a vector propagation method.
   *
   * It computes the distance transform of the input points, and stores the distance to
   * the closest obstacle in each voxel. Also available is the location of the closest point,
   * and the gradient of the field at a point. Expansion of obstacles is performed upto a given
   * radius.
   */

  template<typename PointT>
    class PropagationDistanceField : public DistanceField<PropDistanceFieldVoxel, PointT>
    {

      using DistanceField<PropDistanceFieldVoxel, PointT>::isCellValid;
      using DistanceField<PropDistanceFieldVoxel, PointT>::getCell;
      using DistanceField<PropDistanceFieldVoxel, PointT>::setCell;
      using DistanceField<PropDistanceFieldVoxel, PointT>::num_cells_;
      using DistanceField<PropDistanceFieldVoxel, PointT>::DIM_X;
      using DistanceField<PropDistanceFieldVoxel, PointT>::DIM_Y;
      using DistanceField<PropDistanceFieldVoxel, PointT>::DIM_Z;
      using DistanceField<PropDistanceFieldVoxel, PointT>::initializeVoxelGrid;
      using DistanceField<PropDistanceFieldVoxel, PointT>::getLocationFromCell;
      using DistanceField<PropDistanceFieldVoxel, PointT>::ref;
      using DistanceField<PropDistanceFieldVoxel, PointT>::gridToWorld;
      using DistanceField<PropDistanceFieldVoxel, PointT>::origin_;
      using DistanceField<PropDistanceFieldVoxel, PointT>::size_;

    public:

      /**
       * \brief Constructor for the DistanceField.
       */
      PropagationDistanceField (double size_x, double size_y, double size_z, double resolution, double origin_x, double origin_y, double origin_z,
                                double max_distance);

      PropagationDistanceField (double resolution);

      virtual
      ~PropagationDistanceField ();

      /**
       * \brief Change the set of obstacle points and recalculate the distance field (if there are any changes).
       * \param iterative Calculate the changes in the object voxels, and propogate the changes outward.
       *        Otherwise, clear the distance map and recalculate the entire voxel map.
       */
      virtual void
      updatePointsInField (typename pcl::PointCloud<PointT>::ConstPtr & points, const bool iterative = true);

      /**
       * \brief Add (and expand) a set of points to the distance field.
       */
      virtual void
      addPointsToField (typename pcl::PointCloud<PointT>::ConstPtr & points);

      /**
       * \brief Resets the distance field to the max_distance.
       */
      virtual void
      reset ();

      void
      setInputCloud (typename pcl::PointCloud<PointT>::ConstPtr & cloud);

      void
      getVoxelizedCloud (typename pcl::PointCloud<PointT>::Ptr & cloud)
      {
        cloud = voxelized_cloud_;
      }

      void
      getInputCloud (typename pcl::PointCloud<PointT>::ConstPtr & cloud)
      {
        cloud = cloud_;
      }

      float getDistance(int x, int y, int z);

      /*bool
      isInitialized ()
      {
        return initialized_;
      }*/

      void setDistanceExtend(float d)
      {
        extend_distance_ = d;
      }

      void setHuberSigma(double d)
      {
        huber_sigma_ = d;
      }

      void
      compute ();

      void
      getCorrespondence (const PointT & p, int * idx, float * dist, float sigma, float * color_distance);

      void
      getDerivatives (const PointT & p, Eigen::Vector3f & d);

      VoxelGrid<PropDistanceFieldVoxel> *
      extractVoxelGrid();

      void
      computeFiniteDifferences();

    private:

      typename pcl::PointCloud<PointT>::ConstPtr cloud_;
      typename pcl::PointCloud<PointT>::Ptr voxelized_cloud_;
      pcl::PointCloud<pcl::PointXYZ>::Ptr finite_differences_cloud_;

      /// \brief The set of all the obstacle voxels
      typedef std::set<int3, compareInt3> VoxelSet;
      typedef std::map<int3, int, compareInt3> VoxelMap;
      VoxelSet object_voxel_locations_;
      VoxelMap voxel_locations_to_voxelized_cloud;
      VoxelMap voxel_locations_to_finite_differences_cloud;

      /// \brief Structure used to hold propogation frontier
      std::vector<std::vector<PropDistanceFieldVoxel*> > bucket_queue_;
      double max_distance_;
      unsigned int max_distance_sq_;
      double resolution_;
      float extend_distance_;
      double huber_sigma_;
      std::vector<double> sqrt_table_;

      // neighborhoods:
      // [0] - for expansion of d=0
      // [1] - for expansion of d>=1
      // Under this, we have the 27 directions
      // Then, a list of neighborhoods for each direction
      std::vector<std::vector<std::vector<int3> > > neighborhoods_;

      std::vector<int3> direction_number_to_direction_;

      void initialize(double size_x, double size_y, double size_z, double origin_x, double origin_y, double origin_z, double max_distance);
      void
      addNewObstacleVoxels (const VoxelSet& points);
      void
      removeObstacleVoxels (const VoxelSet& points);
      // starting with the voxels on the queue, propogate values to neighbors up to a certain distance.
      void
      propogate ();
      virtual double
      getDistance (const PropDistanceFieldVoxel& object) const;
      int
      getDirectionNumber (int dx, int dy, int dz) const;
      int3
      getLocationDifference (int directionNumber) const; // TODO- separate out neighborhoods
      void
      initNeighborhoods ();
      static unsigned int
      eucDistSq (int3 point1, int3 point2);

      void
      sign(int x, int y, int z, Eigen::Vector3f & sign_vector);

    };

  ////////////////////////// inline functions follow ////////////////////////////////////////

  inline
  PropDistanceFieldVoxel::PropDistanceFieldVoxel (int distance_sq) :
    distance_square_ (distance_sq)
  {
    closest_point_.x () = PropDistanceFieldVoxel::UNINITIALIZED;
    closest_point_.y () = PropDistanceFieldVoxel::UNINITIALIZED;
    closest_point_.z () = PropDistanceFieldVoxel::UNINITIALIZED;
  }

  inline
  PropDistanceFieldVoxel::PropDistanceFieldVoxel ()
  {
  }

  template<typename PointT>
    inline double
    PropagationDistanceField<PointT>::getDistance (const PropDistanceFieldVoxel& object) const
    {
      return sqrt_table_[object.distance_square_];
    }

  template<typename PointT>
    class SignedPropagationDistanceField : public DistanceField<SignedPropDistanceFieldVoxel, PointT>
    {
      using DistanceField<PropDistanceFieldVoxel, PointT>::isCellValid;
      using DistanceField<PropDistanceFieldVoxel, PointT>::getCell;
      using DistanceField<PropDistanceFieldVoxel, PointT>::num_cells_;
      using DistanceField<PropDistanceFieldVoxel, PointT>::DIM_X;
      using DistanceField<PropDistanceFieldVoxel, PointT>::DIM_Y;
      using DistanceField<PropDistanceFieldVoxel, PointT>::DIM_Z;

    public:

      SignedPropagationDistanceField (double size_x, double size_y, double size_z, double resolution, double origin_x, double origin_y,
                                      double origin_z, double max_distance);
      virtual
      ~SignedPropagationDistanceField ();

      virtual void
      addPointsToField (typename pcl::PointCloud<PointT>::Ptr & points);

      virtual void
      reset ();

    private:
      std::vector<std::vector<SignedPropDistanceFieldVoxel*> > positive_bucket_queue_;
      std::vector<std::vector<SignedPropDistanceFieldVoxel*> > negative_bucket_queue_;
      double max_distance_;
      int max_distance_sq_;

      std::vector<double> sqrt_table_;

      // [0] - for expansion of d=0
      // [1] - for expansion of d>=1
      // Under this, we have the 27 directions
      // Then, a list of neighborhoods for each direction
      std::vector<std::vector<std::vector<int3> > > neighborhoods_;

      std::vector<int3> direction_number_to_direction_;

      virtual double
      getDistance (const SignedPropDistanceFieldVoxel& object) const;
      int
      getDirectionNumber (int dx, int dy, int dz) const;
      void
      initNeighborhoods ();
      static int
      eucDistSq (int3 point1, int3 point2);
    };

  inline
  SignedPropDistanceFieldVoxel::SignedPropDistanceFieldVoxel (int distance_sq_positive, int distance_sq_negative) :
    positive_distance_square_ (distance_sq_positive), negative_distance_square_ (distance_sq_negative),
        closest_positive_point_ (SignedPropDistanceFieldVoxel::UNINITIALIZED), closest_negative_point_ (SignedPropDistanceFieldVoxel::UNINITIALIZED)
  {
  }

  inline
  SignedPropDistanceFieldVoxel::SignedPropDistanceFieldVoxel ()
  {
  }

  template<typename PointT>
  inline double
  SignedPropagationDistanceField<PointT>::getDistance (const SignedPropDistanceFieldVoxel& object) const
  {
    return sqrt_table_[object.positive_distance_square_] - sqrt_table_[object.negative_distance_square_];
  }

}

#endif /* DF_PROPAGATION_DISTANCE_FIELD_H_ */
