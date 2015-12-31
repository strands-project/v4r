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
MeshSource<PointT>::loadOrGenerate (const std::string & model_path, ModelT & model)
{
    const std::string pathmodel = path_ + "/" + model.class_ + "/" + model.id_;

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
            model.view_filenames_ = v4r::io::getFilesInDirectory(pathmodel, ".*view_.*.pcd", false);
            loadInMemorySpecificModel(model);
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
        renderer.setIntrinsics(f, f, cx, cy);
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

        const std::string direc = path_ + "/" + model.class_ + "/" + model.id_ + "/views/";
        v4r::io::createDirIfNotExist(direc);

        for (size_t i = 0; i < model.views_.size (); i++)
        {
            //save generated model for future use
            std::stringstream path_view;
            path_view << direc << "/view_" << i << ".pcd";
            pcl::io::savePCDFileBinary (path_view.str (), *(model.views_[i]));

            std::stringstream path_pose;
            path_pose << direc << "/pose_" << i << ".txt";
            v4r::io::writeMatrixToFile( path_pose.str (), model.poses_[i]);

            std::stringstream path_entropy;
            path_entropy << direc << "/entropy_" << i << ".txt";
            v4r::io::writeFloatToFile (path_entropy.str (), model.self_occlusions_[i]);
        }

        loadOrGenerate ( model_path, model);
    }
}

template<typename PointT>
void
MeshSource<PointT>::loadInMemorySpecificModel(ModelT & model)
{
    const std::string pathmodel = path_ + "/" + model.class_ + "/" + model.id_;

    model.poses_.resize( model.view_filenames_.size() );
    model.views_.resize( model.view_filenames_.size() );
    model.self_occlusions_.resize( model.view_filenames_.size(), 0);

    for (size_t i = 0; i < model.view_filenames_.size (); i++)
    {
        const std::string view_file = pathmodel + "/" + model.view_filenames_[i];
        model.views_[i].reset(new pcl::PointCloud<PointT> ());
        pcl::io::loadPCDFile (view_file, *model.views_[i]);

        std::string pose_fn (view_file);
        boost::replace_last (pose_fn, "view", "pose");
        boost::replace_last (pose_fn, ".pcd", ".txt");
        model.poses_[i] = v4r::io::readMatrixFromFile(pose_fn);

        std::string entropy_fn (view_file);
        boost::replace_last (entropy_fn, "view", "entropy");
        boost::replace_last (entropy_fn, ".pcd", ".txt");
        v4r::io::readFloatFromFile (entropy_fn, model.self_occlusions_[i]);
    }
}

}
