#include "faat_pcl/3d_rec_framework/tools/openni_frame_source.h"
#include <pcl/io/pcd_io.h>
#include <boost/thread/mutex.hpp>
#include <boost/make_shared.hpp>

namespace OpenNIFrameSource
{

  OpenNIFrameSource::OpenNIFrameSource (const std::string& device_id) :
    grabber_ (device_id), most_recent_frame_ (), frame_counter_ (0), active_ (true)
  {
    saved_clouds_ = 0;
    boost::function<void
    (const PointCloudConstPtr&)> frame_cb = boost::bind (&OpenNIFrameSource::onNewFrame, this, _1);
    grabber_.registerCallback (frame_cb);
    grabber_.start ();
  }

  OpenNIFrameSource::~OpenNIFrameSource ()
  {
    // Stop the grabber when shutting down
    grabber_.stop ();
  }

  bool
  OpenNIFrameSource::isActive ()
  {
    return active_;
  }

  const PointCloudPtr
  OpenNIFrameSource::snap ()
  {
    return (most_recent_frame_);
  }

  void
  OpenNIFrameSource::onNewFrame (const PointCloudConstPtr &cloud)
  {
    mutex_.lock ();
    ++frame_counter_;
    most_recent_frame_ = boost::make_shared<PointCloud> (*cloud); // Make a copy of the frame
    mutex_.unlock ();
  }

  void
  OpenNIFrameSource::onKeyboardEvent (const pcl::visualization::KeyboardEvent & event)
  {
    // When the spacebar is pressed, trigger a frame capture
    mutex_.lock ();
    if (event.keyDown ())
    {
      if(event.getKeySym () == "e")
      {
        active_ = false;
      }
      else if(event.getKeySym () == "s")
      {
        if(save_directory_.compare("") == 0)
        {
          PCL_INFO("save directory is empty... set it to something different %s\n", save_directory_.c_str());
          PCL_WARN("Not saving and returning\n");
          mutex_.unlock ();
          return;
        }
        bf::path save_dir = save_directory_;
        if (!bf::exists (save_dir))
        {
          bf::create_directory(save_dir);
        }

        std::stringstream file_name;
        file_name << save_directory_ << "/cloud_" << std::setw(5) << std::setfill('0') << saved_clouds_ << ".pcd";
        std::cout << "saving cloud..." << file_name.str() << std::endl;
        pcl::io::savePCDFileBinary(file_name.str().c_str(), *most_recent_frame_);
        saved_clouds_++;
      }
    }
    mutex_.unlock ();
  }

}
