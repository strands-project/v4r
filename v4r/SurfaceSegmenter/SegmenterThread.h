/*  Copyright (C) 2012
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */

#ifndef V4R_SEGMENT_SEGMENTER_THREAD_H
#define V4R_SEGMENT_SEGMENTER_THREAD_H

#include "Segmenter.h"

#include <pthread.h>
#include <semaphore.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace segment
{

  class SegmenterThread
  {
  private:
    enum Command
    {
      COMPUTE_PLANES, COMPUTE_SURFACES, COMPUTE_OBJECTS, COMPUTE_ALL, STOP, IDLE
    };
    Command command;

    std::string m_object_model_path;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_cloud;
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr m_cloud_labelled;
    pcl::PointCloud<pcl::Normal>::Ptr m_normals;
    std::vector<surface::SurfaceModel::Ptr> m_planes;
    std::vector<surface::SurfaceModel::Ptr> m_surfaces;

    bool cloud_set;
    bool planes_computed;
    bool surfaces_computed;
    bool objects_computed;

    pthread_t m_thread;
    pthread_mutex_t m_mutex_shared;
    sem_t m_compute_start;
    sem_t m_compute_finished;

    friend void*
    ThreadSegmenter (void* c);

  public:
    SegmenterThread (const std::string &object_model_path);
    ~SegmenterThread ();

    /* @brief clears previous results (blocking) */
    void
    clearResults();

    void
    setInputCloud (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

    //    /* @brief compute normals and planes of point-cloud (non-blocking call)
    //     * @brief requires cloud to be set */
    //    void
    //    computePlanes ();
    //
    //    /* @brief compute surfaces of point-cloud (non-blocking call);
    //     * @brief requires planes to be computed */
    //    void
    //    computeSurfaces ();
    //
    //    /* @brief compute normals and planes of point-cloud (non-blocking call)
    //     * @brief requires surfaces to be computed*/
    //    void
    //    computeObjects ();

    /* @brief compute normals and planes, surfaces and objects of point-cloud (non-blocking call) */
    void
    computeAll ();

    /* @brief compute normals and planes, surfaces and objects of point-cloud (blocking call) */
    void
    computeAll (std::vector<surface::SurfaceModel::Ptr> &surfaces);

    /* @brief compute normals and planes, surfaces and objects of point-cloud (blocking call) */
    void
    computeAll (std::vector<surface::SurfaceModel::Ptr> &surfaces, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &labelled);

    //    /* @brief get result of plane segmentation (non-blocking call)
    //     * @return true on success, false if no planes available */
    //    bool
    //    getPlanes (std::vector<surface::SurfaceModel::Ptr> &planes);
    //
    //    /* @brief get result of surface segmentation (non-blocking call)
    //     * @return true on success, false if no surfaces available */
    //    bool
    //    getSurfaces (std::vector<surface::SurfaceModel::Ptr> &surfaces);

    /* @brief get result of object segmentation (non-blocking call)
     * @return true on success, false if no objects available */
    bool
    getObjects (std::vector<surface::SurfaceModel::Ptr> &surfaces);

    /* @brief get result of object segmentation (non-blocking call)
     * @return true on success, false if no objects available */
    bool
    getObjects (std::vector<surface::SurfaceModel::Ptr> &surfaces, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &labelled);

  };

}

#endif
