/*
 * cuda_objectness_3D.h
 *
 *  Created on: Nov 8, 2012
 *      Author: aitor
 */



#ifndef CUDA_OBJECTNESS_3D_H_
#define CUDA_OBJECTNESS_3D_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include "faat_pcl/utils/integral_volume.h"
#include "faat_pcl/segmentation/objectness_3d/objectness_common.h"

#include <boost/shared_ptr.hpp>

#ifdef _MSC_VER
#ifdef FAAT_CUDA_EXPORTS
#define FAAT_CUDA_API __declspec(dllexport)
#else
#define FAAT_CUDA_API __declspec(dllimport)
#endif
#else
#define FAAT_CUDA_API
#endif


namespace faat_pcl
{
  namespace cuda
  {
    namespace segmentation
    {
      class FAAT_CUDA_API Objectness3DCuda
      {

        thrust::device_ptr<int> rivs_;
        thrust::device_ptr<int> rivs_full;
        thrust::device_ptr<int> rivs_occluded;
        thrust::device_ptr<int> rivs_occupancy;
        thrust::device_ptr<int> rivs_occupancy_complete;
        thrust::device_ptr<int> npoints_label_;
        thrust::device_ptr<int> rivs_histograms;
        thrust::device_ptr<int> rivs_color_histograms;
        thrust::device_ptr<int> rivs_npoints_color_histograms;

        int * raw_ptr_rivs;
        int * raw_ptr_rivs_full;
        int * raw_ptr_rivs_occluded;
        int * raw_ptr_rivs_occupancy;
        int * raw_ptr_rivs_rivs_histograms;
        int * raw_ptr_rivs_npoints_label_;
        int * raw_ptr_rivs_rivs_color_histograms;
        int * raw_ptr_rivs_rivs_npoints_color;
        int yuv_hist_size_;

        //thrust::device_ptr<thrust::device_ptr<thrust::device_ptr<int> > > rivs_histograms;
        //thrust::device_vector<thrust::device_vector<thrust::device_ptr<int> > > rivs_histograms;
        float expand_factor_;
        int GRIDSIZE_X_;
        int GRIDSIZE_Y_;
        int GRIDSIZE_Z_;
        int max_label_;
        int table_plane_label_;

        int angle_incr_;
        int min_size_w_;
        int max_size_w_;
        int step_x_, step_y_, step_z_, step_sx_, step_sy_, step_sz_;

        int size_angle_;
        int size_x_;
        int size_y_;
        int size_sx_;
        int size_sy_;

        int start_z_;
        float vpx_, vpy_, vpz_;

        float min_z;
        float max_z;
        float min_x;
        float max_x;
        float min_y;
        float max_y;

        float resolution;

      public:
        Objectness3DCuda (int angle_incr, float expand_factor_, int max_label, int GRID_SIZEX, int GRID_SIZEY, int GRID_SIZEZ, float res, int min_sw, int max_sw,
                            float minx, float maxx, float miny, float maxy, float minz, float maxz);

        ~Objectness3DCuda() {
          cudaFree(raw_ptr_rivs);
          cudaFree(raw_ptr_rivs_npoints_label_);
          cudaFree(raw_ptr_rivs_occluded);
          cudaFree(raw_ptr_rivs_occupancy);
          cudaFree(raw_ptr_rivs_rivs_histograms);
          cudaFree(raw_ptr_rivs_rivs_color_histograms);
          cudaFree(raw_ptr_rivs_rivs_npoints_color);
          cudaFree(raw_ptr_rivs_full);
        }

        void
        setViewpoint(float vx, float vy, float vz) {
          vpx_ = vx;
          vpy_ = vy;
          vpz_ = vz;
        }

        void
        setResolution (float r)
        {
          resolution = r;
        }

        void
        setMinMaxValues (float minx, float maxx, float miny, float maxy, float minz, float maxz)
        {
          min_z = minz;
          max_z = maxz;

          min_y = miny;
          max_y = maxy;

          min_x = minx;
          max_x = maxx;

          std::cout << min_z << " " << min_x << " " << min_y << std::endl;
          std::cout << max_z << " " << max_x << " " << max_y << std::endl;
        }

        void
        addEdgesIV (std::vector<boost::shared_ptr<IntegralVolume> > & rivs);
        void
        addOcclusionVolumes (std::vector<boost::shared_ptr<IntegralVolume> > & rivs);
        void
        addFullVolumes (std::vector<boost::shared_ptr<IntegralVolume> > & rivs);
        void
        addOccupancyVolumes (std::vector<boost::shared_ptr<IntegralVolume> > & rivs);
        void
        addOccupancyCompleteVolumes (std::vector<boost::shared_ptr<IntegralVolume> > & rivs);

        void
        addHistogramVolumes (std::vector<std::vector<boost::shared_ptr<IntegralVolume> > > & rivs);

        void
        addColorHistogramVolumes (std::vector<std::vector<boost::shared_ptr<IntegralVolume> > > & rivs_color,
                                     std::vector< boost::shared_ptr<IntegralVolume> > & rivs_color_points,
                                     int size);

        void
        generateAndComputeBoundingBoxesScore (std::vector<BBox> & boxes, float threshold = 0.75f);

        void
        assign_bb_values(BBox & bb, int idx, float score);

        void
        setNPointsLabel (std::vector<std::vector<int > > & npoints_label);
      };
    }
  }
}

#endif /* CUDA_OBJECTNESS_3D_H_ */
