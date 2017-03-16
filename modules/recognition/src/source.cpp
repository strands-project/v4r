#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/source.h>
#include <glog/logging.h>

namespace v4r
{

template <typename PointT>
Source<PointT>::Source(const std::string &model_database_path, bool has_categories) :
    model_scale_ ( 1.f ),
    load_views_(true),
    compute_normals_(false),
    load_into_memory_(true),
    view_prefix_("cloud_"),
    pose_prefix_ ("pose_"),
    indices_prefix_ ("object_indices_"),
    entropy_prefix_ ("entropy_")
{
    std::vector<std::string> categories;
    if(has_categories)
        categories = io::getFoldersInDirectory(model_database_path);
    else
        categories.push_back("");

    for( const std::string &cat : categories)
    {
        bf::path class_path = model_database_path;
        class_path /= cat;
        std::vector<std::string> instance_names = io::getFoldersInDirectory( class_path.string() );

        LOG(INFO) << "Loading " << instance_names.size() << " object models from folder " << class_path.string() << ". ";
        for(const std::string instance_name : instance_names)
        {
            typename Model<PointT>::Ptr obj (new Model<PointT>);
            obj->id_ = instance_name;
            obj->class_ = cat;

            bf::path object_dir ( class_path.string() );
            object_dir /= instance_name;
            object_dir /= "views";
            const std::string view_pattern = ".*" + view_prefix_ + ".*.pcd";
            std::vector<std::string> training_view_filenames = io::getFilesInDirectory(object_dir.string(), view_pattern, false);

            LOG(INFO) << " ** loading model (class: " << cat << ", instance: " << instance_name << ") with " << training_view_filenames.size() << " views. ";

            for(size_t v_id=0; v_id<training_view_filenames.size(); v_id++)
            {
                typename TrainingView<PointT>::Ptr v (new TrainingView<PointT>);
                bf::path view_filename = object_dir;
                view_filename /= training_view_filenames[v_id];
                v->filename_ = view_filename.string();

                v->pose_filename_ = v->filename_;
                boost::replace_last (v->pose_filename_, view_prefix_, pose_prefix_);
                boost::replace_last (v->pose_filename_, ".pcd", ".txt");

                v->indices_filename_ = v->filename_;
                boost::replace_last (v->indices_filename_, view_prefix_, indices_prefix_);
                boost::replace_last (v->indices_filename_, ".pcd", ".txt");

                obj->addTrainingView( v );
            }

            if(!has_categories)
            {
                bf::path model3D_path ( class_path.string() );
                model3D_path /= instance_name;
                model3D_path /= "3D_model.pcd";
                obj->initialize( model3D_path.string() );
            }
            addModel( obj );
        }
    }
}

template class V4R_EXPORTS Source<pcl::PointXYZ>;
template class V4R_EXPORTS Source<pcl::PointXYZRGB>;

}
