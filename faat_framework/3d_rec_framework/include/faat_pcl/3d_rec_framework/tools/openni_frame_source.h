#ifndef OPENNI_CAPTURE_H
#define OPENNI_CAPTURE_H

#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/filesystem.hpp>
namespace bf=boost::filesystem;

namespace OpenNIFrameSource
{

  typedef pcl::PointXYZRGBA PointT;
  typedef pcl::PointCloud<PointT> PointCloud;
  typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
  typedef pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;

  /* A simple class for capturing data from an OpenNI camera */
  class PCL_EXPORTS OpenNIFrameSource
  {
  public:
    OpenNIFrameSource (const std::string& device_id = "");
    ~OpenNIFrameSource ();

    const PointCloudPtr
    snap ();
    bool
    isActive ();
    void
    onKeyboardEvent (const pcl::visualization::KeyboardEvent & event);

    void
    setSaveDirectory (std::string & dir)
    {
      save_directory_ = dir;
      bf::path save_dir = save_directory_;
      if (bf::exists (save_dir))
      {
        //count the number of elements and reset saved_clouds_ accordingly
        bf::directory_iterator end_iter;
        if (bf::exists (save_dir) && bf::is_directory (save_dir))
        {
          for (bf::directory_iterator dir_iter (save_dir); dir_iter != end_iter; ++dir_iter)
          {
            if (bf::is_regular_file (dir_iter->status ()) && (dir_iter->path ().extension () == ".pcd"))
            {
              std::vector<std::string> strs;
              std::vector<std::string> strs_;
              boost::split (strs, (*dir_iter).path ().string (), boost::is_any_of ("/\\"));
              if (boost::starts_with (strs[strs.size () - 1], "cloud"))
              {
                saved_clouds_++;
              }
            }
          }
        }

        std::cout << "save_clouds_ has been set to" << saved_clouds_ << std::endl;
      }
    }

  protected:
    void
    onNewFrame (const PointCloudConstPtr &cloud);

    pcl::OpenNIGrabber grabber_;
    PointCloudPtr most_recent_frame_;
    int frame_counter_;
    boost::mutex mutex_;
    bool active_;

    //Directory where to save the point clouds
    std::string save_directory_;
    //Counter of save clouds
    int saved_clouds_;
  };

}

#endif
