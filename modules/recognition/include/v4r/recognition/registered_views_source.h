/*
 * ply_source.h
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_REG_VIEWS_SOURCE_H_
#define FAAT_PCL_REC_FRAMEWORK_REG_VIEWS_SOURCE_H_

#include "source.h"
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>

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

template<typename Full3DPointT = pcl::PointXYZRGBNormal, typename PointInT = pcl::PointXYZRGB, typename OutModelPointT = pcl::PointXYZRGB>
class RegisteredViewsSource : public Source<PointInT>
{
    typedef Source<PointInT> SourceT;
    typedef Model<OutModelPointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    using SourceT::path_;
    using SourceT::models_;
    using SourceT::model_scale_;
    using SourceT::load_into_memory_;
    using SourceT::createClassAndModelDirectories;

    std::string model_structure_; //directory with all the views, indices, poses, etc...
    std::string view_prefix_;
    std::string indices_prefix_;
    std::string pose_prefix_;

public:
    RegisteredViewsSource ()
    {
        view_prefix_ = std::string ("cloud");
        pose_prefix_ = std::string("pose");
        indices_prefix_ = std::string("object_indices");
        load_into_memory_ = false;
    }

    void
    setModelStructureDir(const std::string &dir)
    {
        model_structure_ = dir;
    }

    void
    setPrefix (const std::string &pre)
    {
        view_prefix_ = pre;
    }

    void
    assembleModelFromViewsAndPoses(ModelT & model,
                                   std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & poses,
                                   std::vector<pcl::PointIndices> & indices,
                                   typename pcl::PointCloud<PointInT>::Ptr &model_cloud) {
        for(size_t i=0; i < model.views_.size(); i++) {
            Eigen::Matrix4f inv = poses[i];
            inv = inv.inverse();

            typename pcl::PointCloud<PointInT>::Ptr global_cloud_only_indices(new pcl::PointCloud<PointInT>);
            pcl::copyPointCloud(*(model.views_[i]), indices[i], *global_cloud_only_indices);
            typename pcl::PointCloud<PointInT>::Ptr global_cloud(new pcl::PointCloud<PointInT>);
            pcl::transformPointCloud(*global_cloud_only_indices,*global_cloud, inv);
            *(model_cloud) += *global_cloud;
        }
    }

    void
    loadInMemorySpecificModel(const std::string & dir, ModelT & model)
    {
        const std::string pathmodel = dir + "/" + model.class_ + "/" + model.id_;
        if (!v4r::io::existsFolder(pathmodel)) {
            std::cerr << "Training directory " << pathmodel << " does not exist!" << std::endl;
            return;
        }

        for (size_t i = 0; i < model.view_filenames_.size (); i++)
        {
            const std::string view_file = pathmodel + "/" + model.view_filenames_[i];
            typename pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT> ());
            pcl::io::loadPCDFile (view_file, *cloud);

            std::string file_replaced1 (model.view_filenames_[i]);
            boost::replace_all (file_replaced1, view_prefix_, pose_prefix_);
            boost::replace_all (file_replaced1, ".pcd", ".txt");

            //read pose as well
            std::stringstream pose_file;
            pose_file << pathmodel << "/" << file_replaced1;
            Eigen::Matrix4f pose;
            v4r::io::readMatrixFromFile( pose_file.str (), pose);

            //the recognizer assumes transformation from M to CC - i think!
            Eigen::Matrix4f pose_inv = pose.inverse();
            model.poses_.push_back (pose_inv);
            model.self_occlusions_.push_back (-1.f);

            std::string file_replaced2 (model.view_filenames_[i]);
            boost::replace_all (file_replaced2, view_prefix_, indices_prefix_);
            pcl::PointCloud<IndexPoint> obj_indices_cloud;

            std::stringstream oi_file;
            oi_file << pathmodel << "/" << file_replaced2;
            pcl::io::loadPCDFile (oi_file.str(), obj_indices_cloud);
            pcl::PointIndices indices;
            indices.indices.resize(obj_indices_cloud.points.size());
            for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                indices.indices[kk] = obj_indices_cloud.points[kk].idx;

            model.views_.push_back (cloud);
            model.indices_.push_back(indices);
        }
    }

    void
    loadModel (const std::string &model_path, ModelT & model)
    {
        const std::string training_view_path = model_structure_ + "/" + model.class_ + "/" + model.id_;
        const std::string view_pattern = ".*" + view_prefix_ + ".*.pcd";
        v4r::io::getFilesInDirectory(training_view_path, model.view_filenames_, "", view_pattern, false);
        std::cout << "Object class: " << model.class_ << ", id: " << model.id_ << ", views: " << model.view_filenames_.size() << std::endl;

        model.views_.clear();
        model.indices_.clear();
        model.poses_.clear();
        model.self_occlusions_.clear();

        typename pcl::PointCloud<Full3DPointT>::Ptr modell (new pcl::PointCloud<Full3DPointT>);
        typename pcl::PointCloud<Full3DPointT>::Ptr modell_voxelized (new pcl::PointCloud<Full3DPointT>);
        pcl::io::loadPCDFile(model_path, *modell);

        float voxel_grid_size = 0.003f;
        typename pcl::VoxelGrid<Full3DPointT> grid_;
        grid_.setInputCloud (modell);
        grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
        grid_.setDownsampleAllData (true);
        grid_.filter (*modell_voxelized);

        model.normals_assembled_.reset(new pcl::PointCloud<pcl::Normal>);
        model.assembled_.reset (new pcl::PointCloud<PointInT>);

        pcl::copyPointCloud(*modell_voxelized, *model.assembled_);
        pcl::copyPointCloud(*modell_voxelized, *model.normals_assembled_);

        if(load_into_memory_)
        {
            for (size_t i = 0; i < model.view_filenames_.size (); i++)
            {
                const std::string view_file = training_view_path + model.view_filenames_[i];
                typename pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT> ());
                pcl::io::loadPCDFile (view_file, *cloud);

                std::string file_replaced1 (model.view_filenames_[i]);
                boost::replace_all (file_replaced1, view_prefix_, pose_prefix_);
                boost::replace_all (file_replaced1, ".pcd", ".txt");

                const std::string pose_file = training_view_path + "/" + file_replaced1;
                Eigen::Matrix4f pose;
                v4r::io::readMatrixFromFile( pose_file, pose);

                //the recognizer assumes transformation from M to CC - i think!
                Eigen::Matrix4f pose_inv = pose.inverse();

                std::string file_replaced2 (model.view_filenames_[i]);
                boost::replace_all (file_replaced2, view_prefix_, indices_prefix_);
                pcl::PointCloud<IndexPoint> obj_indices_cloud;

                const std::string oi_file = training_view_path + "/" + file_replaced2;
                pcl::io::loadPCDFile (oi_file, obj_indices_cloud);
                pcl::PointIndices indices;
                indices.indices.resize(obj_indices_cloud.points.size());
                for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                    indices.indices[kk] = obj_indices_cloud.points[kk].idx;

                model.views_.push_back (cloud);
                model.indices_.push_back(indices);
                model.poses_.push_back (pose_inv);
                model.self_occlusions_.push_back (-1.f);
            }
        }
    }

    /**
         * \brief Creates the model representation of the training set, generating views if needed
         */
    void
    generate (const std::string &foo)
    {
        //get models in directory
        std::vector < std::string > files;
        v4r::io::getFilesInDirectory (path_, files, "", ".*.pcd",  false);
        std::cout << "There are " << files.size() << " models." << std::endl;

        models_.clear();

        for (size_t i = 0; i < files.size (); i++)
        {
            ModelTPtr m(new ModelT);

            std::vector < std::string > strs;
            boost::split (strs, files[i], boost::is_any_of ("/\\"));
            //            std::string name = strs[strs.size () - 1];

            if (strs.size () == 1)
            {
                m->id_ = strs[0];
            }
            else
            {
                std::stringstream ss;
                for (int j = 0; j < (static_cast<int> (strs.size ()) - 1); j++)
                {
                    ss << strs[j];
                    if (j != (static_cast<int> (strs.size ()) - 1))
                        ss << "/";
                }

                m->class_ = ss.str ();
                m->id_ = strs[strs.size () - 1];
            }

            //check if the model has to be loaded according to the list
            if(!this->isModelIdInList(m->id_))
                continue;

            //check which of them have been trained using training_dir and the model_id_
            //load views, poses and self-occlusions for those that exist
            //generate otherwise

            const std::string model_path = path_ + "/" + files[i];
            loadModel (model_path, *m);

            models_.push_back (m);
        }
    }
};
}

#endif /* REC_FRAMEWORK_MESH_SOURCE_H_ */
