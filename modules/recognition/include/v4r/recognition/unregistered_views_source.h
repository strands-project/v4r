/*
 * ply_source.h
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_UNREG_VIEWS_SOURCE_H_
#define FAAT_PCL_REC_FRAMEWORK_UNREG_VIEWS_SOURCE_H_

#include "source.h"
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "v4r/ORUtils/faat_3d_rec_framework_defines.h"
#include <v4r/utils/filesystem_utils.h>


namespace v4r
{

/**
     * \brief Data source class based on partial views from sensor.
     * In this case, the training data is obtained directly from a depth sensor.
     * The filesystem should contain pcd files (representing a view of an object in
     * camera coordinates) and each view needs to be associated with a txt file
     * containing a 4x4 matrix representing the transformation from camera coordinates
     * to a global object coordinates frame.
     * \author Aitor Aldoma
     */

template<typename PointInT = pcl::PointXYZRGB>
class UnregisteredViewsSource : public Source<PointInT>
{
    typedef Source<PointInT> SourceT;
    typedef Model<PointInT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    using SourceT::path_;
    using SourceT::models_;
    using SourceT::createTrainingDir;
    using SourceT::model_scale_;
    using SourceT::load_into_memory_;
    using SourceT::createClassAndModelDirectories;

    std::string model_structure_; //directory with all the views, indices, poses, etc...
    std::string view_prefix_;
    std::string indices_prefix_;

public:
    UnregisteredViewsSource ()
    {
        view_prefix_ = std::string ("");
        indices_prefix_ = std::string("object_indices_");
        load_into_memory_ = true;
    }

    void
    setModelStructureDir(const std::string &dir)
    {
        model_structure_ = dir;
    }

    void
    setPrefix (const std::string & pre)
    {
        view_prefix_ = pre;
    }

    void
    loadInMemorySpecificModel(const std::string & dir, ModelT & model)
    {
        std::stringstream pathmodel;
        pathmodel << dir << "/" << model.class_ << "/" << model.id_;
        typename pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT> ());
        pcl::io::loadPCDFile (model.view_filenames_[0], *cloud);

        std::string directory, filename;
        char sep = '/';
#ifdef _WIN32
        sep = '\\';
#endif

        size_t position = model.view_filenames_[0].rfind(sep);
        if (position != std::string::npos)
        {
            directory = model.view_filenames_[0].substr(0, position);
            filename = model.view_filenames_[0].substr(position+1, model.view_filenames_[0].length()-1);
        }

        std::stringstream indices_file;
        indices_file << directory << sep << indices_prefix_ << filename;

        pcl::PointCloud<IndexPoint> obj_indices_cloud;

        pcl::io::loadPCDFile (indices_file.str(), obj_indices_cloud);
        pcl::PointIndices indices;
        indices.indices.resize(obj_indices_cloud.points.size());
        for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
            indices.indices[kk] = obj_indices_cloud.points[kk].idx;

        model.views_.push_back (cloud);
        model.indices_.push_back(indices);
        //}
        //}
    }

    void
    loadOrGenerate (const std::string & model_path, ModelT & model)
    {
        model.views_.reset (new std::vector<typename pcl::PointCloud<PointInT>::Ptr>);
        model.indices_.reset (new std::vector<pcl::PointIndices>);

        //load views and poses
        model.view_filenames_.push_back(model_path);

        if(load_into_memory_)
            loadInMemorySpecificModel(model_path, model);
    }

    /**
     * \brief Creates the model representation of the training set, generating views if needed
     */
    void
    generate (const std::string & dummy)
    {
        (void)dummy;
        models_.clear();

        //get models in directory
        std::vector < std::string > folders;
        v4r::getFoldersInDirectory (path_, "", folders);

        for (size_t i = 0; i < folders.size (); i++)
        {
            const std::string class_path = path_ + "/" + folders[i];
            std::vector < std::string > filesInRelFolder;
            v4r::getFilesInDirectory (class_path, filesInRelFolder, "", ".*.pcd", false);
            std::cout << "There are " <<  filesInRelFolder.size() << " files in folder " << folders[i] << ". " << std::endl;

            for (size_t kk = 0; kk < filesInRelFolder.size (); kk++)
            {
                if(filesInRelFolder[kk].length() > indices_prefix_.length())
                {
                    if((filesInRelFolder[kk].compare(0, indices_prefix_.length(), indices_prefix_)==0 ))
                    {
                        std::cout << filesInRelFolder[kk] << " is not a cloud. " << std::endl;
                        continue;
                    }
                }

                ModelTPtr m(new ModelT());
                m->class_ = folders[i];
                m->id_ = filesInRelFolder[kk];

                const std::string model_path = class_path + "/" + filesInRelFolder[kk];
                std::cout << "Calling loadOrGenerate path_model: " << model_path << ", m_class: " << m->class_ << ", m_id: " << m->id_ << std::endl;
                loadOrGenerate (model_path, *m);

                models_.push_back (m);
            }
        }
    }
};
}

#endif /* REC_FRAMEWORK_MESH_SOURCE_H_ */
