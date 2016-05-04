#include <v4r/recognition/registered_views_source.h>
#include <v4r/io/eigen.h>

namespace v4r
{

template<typename PointT>
void
RegisteredViewsSource<PointT>::generate ()
{
    models_.clear();
    std::vector < std::string > model_files = io::getFilesInDirectory (path_, ".*3D_model.pcd",  true);
    std::cout << "There are " << model_files.size() << " models." << std::endl;

    for (const std::string &model_file : model_files)
    {
        ModelTPtr m(new ModelT);

        std::vector < std::string > strs;
        boost::split (strs, model_file, boost::is_any_of ("/\\"));
        //            std::string name = strs[strs.size () - 1];

        if (strs.size () == 2)  // class_name/id_name/3D_model.pcd
        {
            m->id_ = strs[0];
        }
        else if (strs.size()==3)
        {
            m->class_ = strs[0];
            m->id_ = strs[1];
        }
        else
        {
            std::cerr << "Given path " << path_ << " does not have required file structure: (optional: object_class_name)/object_id_name/3D_model.pcd !" << std::endl;
            m->id_ = strs[0];
        }

        //check if the model has to be loaded according to the list
        if(!this->isModelIdInList(m->id_))
            continue;

        //check which of them have been trained using training_dir and the model_id_
        //load views, poses and self-occlusions for those that exist
        //generate otherwise
        loadModel (*m);
        models_.push_back (m);
    }
}

template<typename PointT>
void
RegisteredViewsSource<PointT>::loadModel (ModelT & model)
{
    const std::string training_view_path = path_ + model.class_ + "/" + model.id_ + "/views/";
    const std::string view_pattern = ".*" + view_prefix_ + ".*.pcd";
    model.view_filenames_ = io::getFilesInDirectory(training_view_path, view_pattern, false);
    std::cout << "Object class: " << model.class_ << ", id: " << model.id_ << ", views: " << model.view_filenames_.size() << std::endl;

    typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr modell (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr modell_voxelized (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::io::loadPCDFile(path_ + model.class_ + "/" + model.id_ + "/3D_model.pcd", *modell);

    float voxel_grid_size = 0.003f;
    typename pcl::VoxelGrid<pcl::PointXYZRGBNormal> grid_;
    grid_.setInputCloud (modell);
    grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    grid_.setDownsampleAllData (true);
    grid_.filter (*modell_voxelized);

    model.normals_assembled_.reset(new pcl::PointCloud<pcl::Normal>);
    model.assembled_.reset (new pcl::PointCloud<PointT>);

    pcl::copyPointCloud(*modell_voxelized, *model.assembled_);
    pcl::copyPointCloud(*modell_voxelized, *model.normals_assembled_);

    if(load_into_memory_)
    {
        model.views_.resize( model.view_filenames_.size() );
        model.indices_.resize( model.view_filenames_.size() );
        model.poses_.resize( model.view_filenames_.size() );
        model.self_occlusions_.resize( model.view_filenames_.size() );

        for (size_t i=0; i<model.view_filenames_.size(); i++)
        {
            // load training view
            const std::string view_file = training_view_path + "/" + model.view_filenames_[i];
            model.views_[i].reset( new pcl::PointCloud<PointT> () );
            pcl::io::loadPCDFile (view_file, *model.views_[i]);

            // read pose
            std::string pose_fn (view_file);
            boost::replace_last (pose_fn, view_prefix_, pose_prefix_);
            boost::replace_last (pose_fn, ".pcd", ".txt");
            Eigen::Matrix4f pose = io::readMatrixFromFile( pose_fn );
            model.poses_[i] = pose.inverse(); //the recognizer assumes transformation from M to CC - i think!

            // read object mask
            model.indices_[i].clear();
            std::string obj_indices_fn (view_file);
            boost::replace_last (obj_indices_fn, view_prefix_, indices_prefix_);
            boost::replace_last (obj_indices_fn, ".pcd", ".txt");
            std::ifstream f ( obj_indices_fn.c_str() );
            int idx;
            while (f >> idx)
                model.indices_[i].push_back(idx);
            f.close();

            model.self_occlusions_[i] = -1.f;
        }
    }
    else
    {
        model.views_.clear();
        model.indices_.clear();
        model.poses_.clear();
        model.self_occlusions_.clear();
    }
}


template<typename PointT>
void
RegisteredViewsSource<PointT>::loadInMemorySpecificModel(ModelT &model)
{
    const std::string training_view_path = path_ + "/" + model.class_ + "/" + model.id_ + "/views/";

    if (!v4r::io::existsFolder(training_view_path)) {
        std::cerr << "Training directory " << training_view_path << " does not exist!" << std::endl;
        return;
    }

    model.views_.resize( model.view_filenames_.size() );
    model.indices_.resize( model.view_filenames_.size() );
    model.poses_.resize( model.view_filenames_.size() );
    model.self_occlusions_.resize( model.view_filenames_.size() );

    for (size_t i=0; i<model.view_filenames_.size(); i++)
    {
        // load training view
        const std::string view_file = training_view_path + "/" + model.view_filenames_[i];
        model.views_[i].reset( new pcl::PointCloud<PointT> () );
        pcl::io::loadPCDFile (view_file, *model.views_[i]);

        // read pose
        std::string pose_fn (view_file);
        boost::replace_last (pose_fn, view_prefix_, pose_prefix_);
        boost::replace_last (pose_fn, ".pcd", ".txt");
        Eigen::Matrix4f pose = io::readMatrixFromFile( pose_fn );
        model.poses_[i] = pose.inverse(); //the recognizer assumes transformation from M to CC - i think!

        // read object mask
        model.indices_[i].clear();
        std::string obj_indices_fn (view_file);
        boost::replace_last (obj_indices_fn, view_prefix_, indices_prefix_);
        boost::replace_last (obj_indices_fn, ".pcd", ".txt");
        std::ifstream f ( obj_indices_fn.c_str() );
        int idx;
        while (f >> idx)
            model.indices_[i].push_back(idx);
        f.close();

        model.self_occlusions_[i] = -1.f;
    }
}

template class V4R_EXPORTS RegisteredViewsSource<typename pcl::PointXYZRGB>;
}


