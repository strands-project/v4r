/**
 * $Id$
 *
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
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
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#define AO_DBG_IMAGES

#include <map>
#include <iostream>
#include <fstream>
#include <float.h>
#include <stdexcept>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <v4r/common/convertImage.h>
#include <v4r/common/convertPose.h>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/common/impl/ScopeTime.hpp>
#include <v4r/common/pcl_opencv.h>
#include <v4r/keypoints/ArticulatedObject.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/keypoints/impl/toString.hpp>
#include <v4r/keypoints/io.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r/tracking/ObjectTrackerMono.h>

#include <pcl/io/grabber.h>
//#include <pcl/io/openni2_grabber.h>
#include <pcl/io/openni_grabber.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;
using namespace v4r;

class ObjTrackerMono
{
private:
    std::string log_dir_;
    bool live_;
    int dbg_;

    int ao_sleep_;
    size_t start_;
    size_t last_frame_;
    int camera_id_;
    cv::Mat_<cv::Vec3b> image_;
    cv::Mat_<cv::Vec3b> im_draw_;

    cv::Mat_<double> dist_coeffs_;
    cv::Mat_<double> intrinsic_;
    cv::Mat_<double> src_dist_coeffs_;
    cv::Mat_<double> src_intrinsic_;
    Eigen::Matrix4f pose_;
    v4r::ArticulatedObject::Ptr model_;

    std::string model_file_, filenames_, cam_file_, cam_file_model_;
    std::vector<Eigen::Vector3f> cam_trajectory;
    bool log_cams_;
    bool log_files_;
    bool loop_;
    v4r::ObjectTrackerMono::Parameter param_;

    boost::shared_ptr<pcl::Grabber> interface_;
    boost::shared_ptr<cv::VideoCapture> cap_;
    v4r::ObjectTrackerMono::Ptr tracker_;

    /**
 * @brief drawCoordinateSystem
 * @param im
 * @param pose
 * @param intrinsic
 * @param dist_coeffs
 */
    void drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &pose, const cv::Mat_<double> &intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness) const
    {
        Eigen::Matrix3f R = pose.topLeftCorner<3,3>();
        Eigen::Vector3f t = pose.block<3, 1>(0,3);

        Eigen::Vector3f pt0  = R * Eigen::Vector3f(0,0,0) + t;
        Eigen::Vector3f pt_x = R * Eigen::Vector3f(size,0,0) + t;
        Eigen::Vector3f pt_y = R * Eigen::Vector3f(0,size,0) + t;
        Eigen::Vector3f pt_z = R * Eigen::Vector3f(0,0,size) +t ;

        cv::Point2f im_pt0, im_pt_x, im_pt_y, im_pt_z;

        if (!dist_coeffs.empty())
        {
            v4r::projectPointToImage(&pt0 [0], &intrinsic(0), &dist_coeffs(0), &im_pt0.x );
            v4r::projectPointToImage(&pt_x[0], &intrinsic(0), &dist_coeffs(0), &im_pt_x.x);
            v4r::projectPointToImage(&pt_y[0], &intrinsic(0), &dist_coeffs(0), &im_pt_y.x);
            v4r::projectPointToImage(&pt_z[0], &intrinsic(0), &dist_coeffs(0), &im_pt_z.x);
        }
        else
        {
            v4r::projectPointToImage(&pt0 [0], &intrinsic(0), &im_pt0.x );
            v4r::projectPointToImage(&pt_x[0], &intrinsic(0), &im_pt_x.x);
            v4r::projectPointToImage(&pt_y[0], &intrinsic(0), &im_pt_y.x);
            v4r::projectPointToImage(&pt_z[0], &intrinsic(0), &im_pt_z.x);
        }

        cv::line(im, im_pt0, im_pt_x, CV_RGB(255,0,0), thickness);
        cv::line(im, im_pt0, im_pt_y, CV_RGB(0,255,0), thickness);
        cv::line(im, im_pt0, im_pt_z, CV_RGB(0,0,255), thickness);
    }

    /**
 * drawConfidenceBar
 */
    void drawConfidenceBar(cv::Mat &im, const double &conf) const
    {
        int bar_start = 50, bar_end = 200;
        int diff = bar_end-bar_start;
        int draw_end = diff*conf;
        double col_scale = 255./(double)diff;
        cv::Point2f pt1(0,30);
        cv::Point2f pt2(0,30);
        cv::Vec3b col(0,0,0);

        if (draw_end<=0)
            draw_end = 1;

        for (int i=0; i<draw_end; i++)
        {
            col = cv::Vec3b(255-(i*col_scale), i*col_scale, 0);
            pt1.x = bar_start+i;
            pt2.x = bar_start+i+1;
            cv::line(im, pt1, pt2, CV_RGB(col[0],col[1],col[2]), 8);
        }
    }


public:
    ObjTrackerMono()
    {
        log_dir_ = "/tmp/log/";
        live_ = -1;
        dbg_ = 0;
        ao_sleep_ = 1;
        start_ = 0;
        last_frame_ = 100;
        loop_ = false;
        camera_id_ = -1;

        intrinsic_ = cv::Mat_<double>::eye(3,3);
        dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);
        log_cams_ = false;
        log_files_ = false;

        param_.kt_param.plk_param.use_ncc = true;
        param_.kt_param.plk_param.ncc_residual = .5;

        tracker_.reset(new v4r::ObjectTrackerMono(param_));
        model_.reset(new ArticulatedObject());
    }

    void
    cloud_cb (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
    {
        image_ = v4r::ConvertPCLCloud2Image(*cloud);
    }


    void setup(int argc, char **argv)
    {
        po::options_description desc("Monocular Object Tracker\n======================================\n**Allowed options");
        desc.add_options()
                ("help,h", "produce help message")
                ("model_file,m", po::value<std::string>(&model_file_)->required(), "model file (.ao)  (pointcloud or stored model with .bin)")
                ("live,l", po::bool_switch(&live_), "use live stream of camera id")
                ("camera_id,c", po::value<int>(&camera_id_)->default_value(camera_id_), "camera id (-1... use PCL Openni)")
                ("start frame,s", po::value<size_t>(&start_)->default_value(start_), "start frame")
                ("last frame,e", po::value<size_t>(&last_frame_)->default_value(last_frame_), "last frame")
                ("cam_file,a", po::value<std::string>(&cam_file_), "camera calibration files for the tracking sequence (.yml) (opencv format)")
                ("filenames,f", po::value<std::string>(&filenames_), "image filename (printf-style) or filename")
                ("log_dir", po::value<std::string>(&log_dir_)->default_value(log_dir_), "directory for logging output")
                ("loop", po::bool_switch(&loop_), "loop through files")
                ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
        }

        try
        {
            po::notify(vm);
        }
        catch(std::exception& e)
        {
            std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        }

        if (live_)
        {
            loop_ = true;

            if(camera_id_ == -1)
            {
                try
                {
//                    interface_.reset( new pcl::io::OpenNI2Grabber() );
                    interface_.reset( new pcl::OpenNIGrabber() );
                    boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f = boost::bind (&ObjTrackerMono::cloud_cb, this, _1);
                    interface_->registerCallback (f);
                }
                catch (pcl::IOException e)
                {
                    std::cerr << "PCL threw error " << e.what() << ". Could not start camera..." << std::endl;
                }
            }
            else
            {
                cap_.reset(new cv::VideoCapture);
                cap_->open(camera_id_);
                cap_->set(CV_CAP_PROP_FRAME_WIDTH, 640);
                cap_->set(CV_CAP_PROP_FRAME_HEIGHT, 480);
                if( !cap_->isOpened() )
                    throw std::runtime_error ("Could not initialize capturing...");
            }
        }

        intrinsic_(0,0)=intrinsic_(1,1)=525;
        intrinsic_(0,2)=320, intrinsic_(1,2)=240;

        if (!cam_file_.empty())
        {
            cv::FileStorage fs( cam_file_, cv::FileStorage::READ );
            fs["camera_matrix"] >> intrinsic_;
            fs["distortion_coefficients"] >> dist_coeffs_;
        }
        if (!cam_file_model_.empty())
        {
            cv::FileStorage fs( cam_file_model_, cv::FileStorage::READ );
            fs["camera_matrix"] >> src_intrinsic_;
            fs["distortion_coefficients"] >> src_dist_coeffs_;
        }

        if(!src_intrinsic_.empty())
            tracker_->setObjectCameraParameter(src_intrinsic_, src_dist_coeffs_);

        tracker_->setCameraParameter(intrinsic_, dist_coeffs_);

        // -------------------- load model ---------------------------
        std::cout << "Load: "<< model_file_ << std::endl;

        if(v4r::io::read(model_file_, model_))
            tracker_->setObjectModel(model_);
        else
            throw std::runtime_error("Tracking model file not found!");
    }

    void run()
    {
        double time, conf;
        double mean_time=0;
        int cnt4time=0;

        cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );

        if(camera_id_ == -1)
                interface_->start ();

        for (size_t i=start_; i<last_frame_ || loop_; i++)
        {
            std::cout << "---------------- FRAME #" << i << " -----------------------" << std::endl;

            if (live_){
                if(camera_id_ != -1)
                    *cap_ >> image_;
            }
            else {
                char filename[PATH_MAX];
                snprintf(filename,PATH_MAX, filenames_.c_str(), i);
                std::cout << filename << std::endl;
                image_ = cv::imread(filename, 1);
            }

            if (image_.empty())
            {
                cv::waitKey(200);
                continue;
            }

            image_.copyTo(im_draw_);

            bool is_ok = false;

            if (dbg_)
                tracker_->dbg = im_draw_;

            {
                v4r::ScopeTime t("overall time");
                is_ok = tracker_->track(image_, pose_, conf);
                time = t.getTime();
            }

            cnt4time++;
            mean_time += time;
            std::cout << "tracker conf = " << conf << std::endl
                      << "mean_time = " << mean_time/double(cnt4time) << " (" << cnt4time << ")" << std::endl;

            if (is_ok)
                std::cout << "Status: stable tracking" << std::endl;
            else if (conf > 0.05)
                std::cout << "Status: unstable tracking" << std::endl;
            else
                std::cout << "Status: object lost!" << std::endl;

            if (conf>0.05 && i>start_)
            {
                if (log_cams_)
                    cam_trajectory.push_back(pose_.block<3, 1>(0,3));

                drawCoordinateSystem(im_draw_, pose_, intrinsic_, dist_coeffs_, 0.1, 4);
            }

            drawConfidenceBar(im_draw_, conf);

            cv::imshow("image", im_draw_);

            int key = cv::waitKey(ao_sleep_);
            if (((char)key)==27) break;
            if (((char)key)=='r') ao_sleep_=1;
            if (((char)key)=='s') ao_sleep_=0;
            if (((char)key)=='c') cam_trajectory.clear();
            if (((char)key)=='l') { log_cams_ = !log_cams_; cv::waitKey(100); }
            if (((char)key)=='r') { tracker_->reset(); }
        }

        if(camera_id_ == -1)
                interface_->stop ();
    }
};


int main(int argc, char *argv[] )
{
    ObjTrackerMono ot;
    ot.setup(argc, argv);
    ot.run();
    return 0;
}
