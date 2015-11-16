#include <v4r/recognition/mesh_source.h>
#include <v4r/recognition/vtk_model_sampling.h>

namespace v4r
{

template<>
pcl::PointCloud<pcl::PointXYZRGB>
MeshSource<pcl::PointXYZRGB>::renderCloud (const DepthmapRenderer &renderer, float &visible)
{
    return renderer.renderPointcloudColor(visible);
}

template<typename PointT>
pcl::PointCloud<PointT>
MeshSource<PointT>::renderCloud (const DepthmapRenderer &renderer, float &visible)
{
    return renderer.renderPointcloud(visible);
}

template<typename PointT>
void
MeshSource<PointT>::loadOrGenerate (const std::string & dir, const std::string & model_path, ModelT & model)
{
    const std::string pathmodel = dir + "/" + model.class_ + "/" + model.id_;

    model.views_.clear();
    model.poses_.clear();
    model.self_occlusions_.clear();
    model.assembled_.reset (new pcl::PointCloud<PointT>);
    uniform_sampling (model_path, 100000, *model.assembled_, model_scale_);

    if(compute_normals_) {
        std::cout << "Computing normals..." << std::endl;
        model.computeNormalsAssembledCloud(radius_normals_);
    }

    if (v4r::io::existsFolder(pathmodel))
    {
        if(load_into_memory_) {
            v4r::io::getFilesInDirectory(pathmodel, model.view_filenames_, "", ".*view_.*.pcd", false);
            loadInMemorySpecificModel(dir, model);
        }
    }
    else
    {
        int img_width = resolution_;
        int img_height = resolution_;

        // To preserve Kinect camera parameters (640x480 / f=525)
        const float f = 150.f;
        const float cx = img_width / 2.f;
        const float cy = img_height / 2.f;
        DepthmapRenderer renderer(img_width, img_height);
        renderer.setIntrinsics(f,f,cx,cy);
        DepthmapRendererModel rmodel(model_path);
        renderer.setModel(&rmodel);

        std::vector<Eigen::Vector3f> sphere = renderer.createSphere(radius_sphere_, tes_level_);

        for(size_t i=0; i<sphere.size(); i++){
            //get point from list
            Eigen::Vector3f point = sphere[i];
            //get a camera pose looking at the center:
            Eigen::Matrix4f orientation = renderer.getPoseLookingToCenterFrom(point);
            renderer.setCamPose(orientation);
            float visible;
            typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>(renderCloud(renderer, visible)));
            const Eigen::Matrix4f tf = v4r::RotTrans2Mat4f(cloud->sensor_orientation_, cloud->sensor_origin_);

            // reset view point otherwise pcl visualization is potentially messed up
            Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            cloud->sensor_origin_ = zero_origin;

            if(!gen_organized_)   // remove nan points from cloud
            {
                size_t kept=0;
                for(size_t idx=0; idx<cloud->points.size(); idx++)
                {
                    const PointT &pt = cloud->points[idx];
                    if ( pcl::isFinite(pt) )
                        cloud->points[kept++] = pt;
                }
                cloud->points.resize(kept);
                cloud->width = kept;
                cloud->height = 1;
            }

            model.views_.push_back (cloud);
            model.poses_.push_back (tf);
            model.self_occlusions_.push_back (0); // NOT IMPLEMENTED
        }

        std::stringstream direc; direc << dir << "/" << model.class_ << "/" << model.id_;
        this->createClassAndModelDirectories (dir, model.class_, model.id_);

        for (size_t i = 0; i < model.views_.size (); i++)
        {
            //save generated model for future use
            std::stringstream path_view;
            path_view << direc.str () << "/view_" << i << ".pcd";
            pcl::io::savePCDFileBinary (path_view.str (), *(model.views_[i]));

            std::stringstream path_pose;
            path_pose << direc.str () << "/pose_" << i << ".txt";

            v4r::io::writeMatrixToFile( path_pose.str (), model.poses_[i]);

            std::stringstream path_entropy;
            path_entropy << direc.str () << "/entropy_" << i << ".txt";
            v4r::io::writeFloatToFile (path_entropy.str (), model.self_occlusions_[i]);
        }

        loadOrGenerate (dir, model_path, model);
    }
}

template<typename PointT>
void
MeshSource<PointT>::loadInMemorySpecificModel(const std::string & dir, ModelT & model)
{
    const std::string pathmodel = dir + "/" + model.class_ + "/" + model.id_;

    for (size_t i = 0; i < model.view_filenames_.size (); i++)
    {
        const std::string view_file = pathmodel + "/" + model.view_filenames_[i];
        typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT> ());
        pcl::io::loadPCDFile (view_file, *cloud);

        model.views_.push_back (cloud);

        std::string file_replaced1 (model.view_filenames_[i]);
        boost::replace_all (file_replaced1, "view", "pose");
        boost::replace_all (file_replaced1, ".pcd", ".txt");

        std::string file_replaced2 (model.view_filenames_[i]);
        boost::replace_all (file_replaced2, "view", "entropy");
        boost::replace_all (file_replaced2, ".pcd", ".txt");

        //read pose as well
        const std::string pose_file = pathmodel + "/" + file_replaced1;

        Eigen::Matrix4f pose;
        v4r::io::readMatrixFromFile(pose_file, pose);

        model.poses_.push_back (pose);

        //read entropy as well
        const std::string entropy_file = pathmodel + "/" + file_replaced2;
        float entropy = 0;
        v4r::io::readFloatFromFile (entropy_file, entropy);
        model.self_occlusions_.push_back (entropy);
    }
}

}
