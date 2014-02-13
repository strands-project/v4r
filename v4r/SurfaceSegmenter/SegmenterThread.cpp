#include "SegmenterThread.h"
#include <pcl/common/io.h>

using namespace segment;

namespace segment
{
  void*
  ThreadSegmenter (void* c)
  {
    SegmenterThread *st = (SegmenterThread*)c;

    segment::Segmenter segmenter (st->m_object_model_path);
    segmenter.setDetail(2);
    printf ("[ThreadSegmenter] model_path: '%s'\n", st->m_object_model_path.c_str ());

    bool stop = false;

    while (!stop)
    {
      printf ("[ThreadSegmenter] SegmenterThread ... waiting for command\n");
      sem_wait (&st->m_compute_start);

      switch (st->command)
      {
      /**********************************************************
       * NOTE MZ_: does not compile right now after necessary interface changes to Segmenter.
       *********************************************************/
        case SegmenterThread::COMPUTE_PLANES:
          printf ("[ThreadSegmenter] COMPUTE_PLANES\n");
          pthread_mutex_lock (&st->m_mutex_shared);
          segmenter.computeNormals (st->m_cloud, st->m_normals);
          segmenter.computePlanes (st->m_cloud, st->m_normals, st->m_planes);
          st->planes_computed = true;
          st->command = SegmenterThread::IDLE;
          pthread_mutex_unlock (&st->m_mutex_shared);
          break;

        case SegmenterThread::COMPUTE_SURFACES:
          printf ("[ThreadSegmenter] COMPUTE_SURFACES\n");
          pthread_mutex_lock (&st->m_mutex_shared);
          segmenter.computeSurfaces (st->m_cloud, st->m_normals, st->m_surfaces);
          st->surfaces_computed = true;
          st->command = SegmenterThread::IDLE;
          pthread_mutex_unlock (&st->m_mutex_shared);
          break;

        case SegmenterThread::COMPUTE_OBJECTS:
          printf ("[ThreadSegmenter] COMPUTE_OBJECTS\n");
          pthread_mutex_lock (&st->m_mutex_shared);
          segmenter.computeObjects (st->m_cloud, st->m_surfaces, st->m_cloud_labelled);
          st->objects_computed = true;
          st->command = SegmenterThread::IDLE;
          pthread_mutex_unlock (&st->m_mutex_shared);
          break;

        case SegmenterThread::COMPUTE_ALL:
          printf ("[ThreadSegmenter] COMPUTE_ALL\n");
          pthread_mutex_lock (&st->m_mutex_shared);

          printf ("\n[ThreadSegmenter] PLANE-SEGMENTATION\n\n");
          segmenter.computeNormals (st->m_cloud, st->m_normals);
          segmenter.computePlanes (st->m_cloud, st->m_normals, st->m_planes);
          st->planes_computed = true;

          printf ("\n[ThreadSegmenter] SURFACE-SELECTION\n\n");
          st->m_surfaces.resize (st->m_planes.size ());
          for (size_t i = 0; i < st->m_planes.size (); i++)
          {
            st->m_surfaces[i].reset (new surface::SurfaceModel);
            *st->m_surfaces[i] = *st->m_planes[i];
          }
          segmenter.computeSurfaces (st->m_cloud, st->m_normals, st->m_surfaces);
          st->surfaces_computed = true;

          printf ("\n[ThreadSegmenter] OBJECT-CLUSTERING\n\n");
          segmenter.computeObjects (st->m_cloud, st->m_surfaces, st->m_cloud_labelled);
          st->objects_computed = true;

          st->command = SegmenterThread::IDLE;
          pthread_mutex_unlock (&st->m_mutex_shared);
          break;

        case SegmenterThread::STOP:
          printf ("[ThreadSegmenter] STOP\n");
          stop = true;
          break;

        default:
          printf ("[ThreadSegmenter] default\n");
          break;
      }

      sem_post (&st->m_compute_finished);
    }

    printf("[SegmenterThread::ThreadSegmenter] finished\n");

    return (void*)0;
  }
}

SegmenterThread::SegmenterThread (const std::string &object_model_path) :
  cloud_set (false), planes_computed (false), surfaces_computed (false), objects_computed (false)
{
  pthread_mutex_init (&m_mutex_shared, NULL);
  sem_init (&m_compute_start, 0, 0);
  sem_init (&m_compute_finished, 0, 0);

  command = IDLE;
  m_cloud.reset (new pcl::PointCloud<pcl::PointXYZRGB>);
  m_cloud_labelled.reset (new pcl::PointCloud<pcl::PointXYZRGBL>);
  m_normals.reset (new pcl::PointCloud<pcl::Normal>);

  m_object_model_path = object_model_path;

  pthread_create (&m_thread, NULL, ThreadSegmenter, this);
}

SegmenterThread::~SegmenterThread ()
{
  printf("[SegmenterThread::~SegmenterThread] A\n");
  // stop
  pthread_mutex_lock (&m_mutex_shared);
  command = STOP;
  pthread_mutex_unlock (&m_mutex_shared);
  sem_post (&m_compute_start);
  sem_wait (&m_compute_finished);

  // destroy
  pthread_join (m_thread, NULL);
  sem_destroy (&m_compute_start);
  sem_destroy (&m_compute_finished);
  pthread_mutex_destroy (&m_mutex_shared);
  printf("[SegmenterThread::~SegmenterThread] B\n");
}

void
SegmenterThread::clearResults ()
{
  pthread_mutex_lock (&m_mutex_shared);
  m_cloud.reset (new pcl::PointCloud<pcl::PointXYZRGB>);
  m_cloud_labelled.reset (new pcl::PointCloud<pcl::PointXYZRGBL>);
  m_normals.reset (new pcl::PointCloud<pcl::Normal>);
  m_planes.clear ();
  m_surfaces.clear ();
  cloud_set = false;
  planes_computed = false;
  surfaces_computed = false;
  objects_computed = false;
  pthread_mutex_unlock (&m_mutex_shared);
}

void
SegmenterThread::setInputCloud (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
  pthread_mutex_lock (&m_mutex_shared);
  pcl::copyPointCloud (*cloud, *m_cloud);
  pcl::copyPointCloud (*cloud, *m_cloud_labelled);
  m_planes.clear ();
  m_surfaces.clear ();
  cloud_set = true;
  planes_computed = false;
  surfaces_computed = false;
  objects_computed = false;
  pthread_mutex_unlock (&m_mutex_shared);
}

//void
//SegmenterThread::computePlanes ()
//{
//  pthread_mutex_lock (&m_mutex_shared);
//  if (cloud_set)
//  {
//    command = COMPUTE_PLANES;
//    sem_post (&m_compute_start);
//  }
//  else
//  {
//    printf ("[SegmenterThread::computePlanes] Warning, cloud not set. Skipping\n");
//  }
//  pthread_mutex_unlock (&m_mutex_shared);
//}
//
//void
//SegmenterThread::computeSurfaces ()
//{
//  pthread_mutex_lock (&m_mutex_shared);
//  if (planes_computed)
//  {
//    // deep copy of planes
//    m_surfaces.resize (m_planes.size ());
//    for (size_t i = 0; i < m_planes.size (); i++)
//    {
//      m_surfaces[i].reset (new surface::SurfaceModel);
//      *m_surfaces[i] = *m_planes[i];
//    }
//
//    command = COMPUTE_SURFACES;
//    sem_post (&m_compute_start);
//  }
//  else
//  {
//    printf ("[SegmenterThread::computeSurfaces] Warning, planes not computed. Skipping\n");
//  }
//  pthread_mutex_unlock (&m_mutex_shared);
//}
//
//void
//SegmenterThread::computeObjects ()
//{
//  pthread_mutex_lock (&m_mutex_shared);
//  if (surfaces_computed)
//  {
//    command = COMPUTE_OBJECTS;
//    sem_post (&m_compute_start);
//  }
//  else
//  {
//    printf ("[SegmenterThread::computeObjects] Warning, planes not computed. Skipping\n");
//  }
//  pthread_mutex_unlock (&m_mutex_shared);
//}

void
SegmenterThread::computeAll ()
{
  if (!cloud_set)
  {
    printf ("[SegmenterThread::computeAll] Warning, cloud not set. Skipping\n");
    return;
  }

  pthread_mutex_lock (&m_mutex_shared);
  command = COMPUTE_ALL;
  sem_post (&m_compute_start);
  pthread_mutex_unlock (&m_mutex_shared);
}

void
SegmenterThread::computeAll (std::vector<surface::SurfaceModel::Ptr> &surfaces)
{
  if (!cloud_set)
  {
    printf ("[SegmenterThread::computeAll] Warning, cloud not set. Skipping\n");
    return;
  }

  pthread_mutex_lock (&m_mutex_shared);
  command = COMPUTE_ALL;
  pthread_mutex_unlock (&m_mutex_shared);

  sem_post (&m_compute_start);
  sem_wait (&m_compute_finished);

  pthread_mutex_lock (&m_mutex_shared);
  // deep copy
  surfaces.resize (m_surfaces.size ());
  for (size_t i = 0; i < surfaces.size (); i++)
  {
    surfaces[i].reset (new surface::SurfaceModel);
    *surfaces[i] = *m_surfaces[i];
  }
  pthread_mutex_unlock (&m_mutex_shared);
}

void
SegmenterThread::computeAll (std::vector<surface::SurfaceModel::Ptr> &surfaces,
                             pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &labelled)
{
  if (!cloud_set)
  {
    printf ("[SegmenterThread::computeAll] Warning, cloud not set. Skipping\n");
    return;
  }

  pthread_mutex_lock (&m_mutex_shared);
  command = COMPUTE_ALL;
  pthread_mutex_unlock (&m_mutex_shared);

  sem_post (&m_compute_start);
  sem_wait (&m_compute_finished);

  pthread_mutex_lock (&m_mutex_shared);
  // deep copy
  surfaces.resize (m_surfaces.size ());
  for (size_t i = 0; i < surfaces.size (); i++)
  {
    surfaces[i].reset (new surface::SurfaceModel);
    *surfaces[i] = *m_surfaces[i];
  }
  pcl::copyPointCloud (*m_cloud_labelled, *labelled);
  pthread_mutex_unlock (&m_mutex_shared);
}

//bool
//SegmenterThread::getPlanes (std::vector<surface::SurfaceModel::Ptr> &planes)
//{
//  if (!planes_computed)
//    return false;
//
//  pthread_mutex_lock (&m_mutex_shared);
//  // deep copy
//  planes.resize (m_planes.size ());
//  for (size_t i = 0; i < planes.size (); i++)
//  {
//    planes[i].reset (new surface::SurfaceModel);
//    *planes[i] = *m_planes[i];
//  }
//  pthread_mutex_unlock (&m_mutex_shared);
//
//  return true;
//}
//
//bool
//SegmenterThread::getSurfaces (std::vector<surface::SurfaceModel::Ptr> &surfaces)
//{
//  if (!surfaces_computed)
//    return false;
//
//  pthread_mutex_lock (&m_mutex_shared);
//  // deep copy
//  surfaces.resize (m_surfaces.size ());
//  for (size_t i = 0; i < surfaces.size (); i++)
//  {
//    surfaces[i].reset (new surface::SurfaceModel);
//    *surfaces[i] = *m_surfaces[i];
//  }
//  pthread_mutex_unlock (&m_mutex_shared);
//
//  return true;
//}

bool
SegmenterThread::getObjects (std::vector<surface::SurfaceModel::Ptr> &surfaces)
{
  if (!objects_computed)
    return false;

  pthread_mutex_lock (&m_mutex_shared);
  // deep copy
  surfaces.resize (m_surfaces.size ());
  for (size_t i = 0; i < surfaces.size (); i++)
  {
    surfaces[i].reset (new surface::SurfaceModel);
    *surfaces[i] = *m_surfaces[i];
  }
  pthread_mutex_unlock (&m_mutex_shared);

  return true;
}

bool
SegmenterThread::getObjects (std::vector<surface::SurfaceModel::Ptr> &surfaces,
                             pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &labelled)
{
  if (!objects_computed)
    return false;

  pthread_mutex_lock (&m_mutex_shared);

  // deep copy
  surfaces.resize (m_surfaces.size ());
  for (size_t i = 0; i < surfaces.size (); i++)
  {
    surfaces[i].reset (new surface::SurfaceModel);
    *surfaces[i] = *m_surfaces[i];
  }

  pcl::copyPointCloud (*m_cloud_labelled, *labelled);

  pthread_mutex_unlock (&m_mutex_shared);

  return true;
}

