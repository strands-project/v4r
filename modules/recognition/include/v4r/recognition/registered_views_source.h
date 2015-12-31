/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#ifndef V4R_REG_VIEWS_SOURCE_H_
#define V4R_REG_VIEWS_SOURCE_H_

#include "source.h"
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/core/macros.h>
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
class V4R_EXPORTS RegisteredViewsSource : public Source<PointInT>
{
    typedef Source<PointInT> SourceT;
    typedef Model<OutModelPointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    using SourceT::path_;
    using SourceT::models_;
    using SourceT::load_into_memory_;
    using SourceT::resolution_;

    std::string view_prefix_;
    std::string indices_prefix_;
    std::string pose_prefix_;

public:
    RegisteredViewsSource (float resolution = 0.001f)
    {
        resolution_ = resolution;
        view_prefix_ = std::string ("cloud");
        pose_prefix_ = std::string("pose");
        indices_prefix_ = std::string("object_indices");
        load_into_memory_ = false;
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
                                   typename pcl::PointCloud<PointInT>::Ptr &model_cloud)
    {
        for(size_t i=0; i < model.views_.size(); i++)
        {
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
    loadInMemorySpecificModel(ModelT &model)
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
            model.views_[i].reset( new pcl::PointCloud<PointInT> () );
            pcl::io::loadPCDFile (view_file, *model.views_[i]);

            // read pose
            std::string pose_fn (view_file);
            boost::replace_last (pose_fn, view_prefix_, pose_prefix_);
            boost::replace_last (pose_fn, ".pcd", ".txt");
            Eigen::Matrix4f pose = io::readMatrixFromFile( pose_fn );
            model.poses_[i] = pose.inverse(); //the recognizer assumes transformation from M to CC - i think!

            // read object mask
            model.indices_[i].indices.clear();
            std::string obj_indices_fn (view_file);
            boost::replace_last (obj_indices_fn, view_prefix_, indices_prefix_);
            boost::replace_last (obj_indices_fn, ".pcd", ".txt");
            std::ifstream f ( obj_indices_fn.c_str() );
            int idx;
            while (f >> idx)
                model.indices_[i].indices.push_back(idx);
            f.close();

            model.self_occlusions_[i] = -1.f;
        }
    }

    void
    loadModel (ModelT & model)
    {
        const std::string training_view_path = path_ + model.class_ + "/" + model.id_ + "/views/";
        const std::string view_pattern = ".*" + view_prefix_ + ".*.pcd";
        model.view_filenames_ = io::getFilesInDirectory(training_view_path, view_pattern, false);
        std::cout << "Object class: " << model.class_ << ", id: " << model.id_ << ", views: " << model.view_filenames_.size() << std::endl;

        typename pcl::PointCloud<Full3DPointT>::Ptr modell (new pcl::PointCloud<Full3DPointT>);
        typename pcl::PointCloud<Full3DPointT>::Ptr modell_voxelized (new pcl::PointCloud<Full3DPointT>);
        pcl::io::loadPCDFile(path_ + model.class_ + "/" + model.id_ + "/3D_model.pcd", *modell);

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
            model.views_.resize( model.view_filenames_.size() );
            model.indices_.resize( model.view_filenames_.size() );
            model.poses_.resize( model.view_filenames_.size() );
            model.self_occlusions_.resize( model.view_filenames_.size() );

            for (size_t i=0; i<model.view_filenames_.size(); i++)
            {
                // load training view
                const std::string view_file = training_view_path + "/" + model.view_filenames_[i];
                model.views_[i].reset( new pcl::PointCloud<PointInT> () );
                pcl::io::loadPCDFile (view_file, *model.views_[i]);

                // read pose
                std::string pose_fn (view_file);
                boost::replace_last (pose_fn, view_prefix_, pose_prefix_);
                boost::replace_last (pose_fn, ".pcd", ".txt");
                Eigen::Matrix4f pose = io::readMatrixFromFile( pose_fn );
                model.poses_[i] = pose.inverse(); //the recognizer assumes transformation from M to CC - i think!

                // read object mask
                model.indices_[i].indices.clear();
                std::string obj_indices_fn (view_file);
                boost::replace_last (obj_indices_fn, view_prefix_, indices_prefix_);
                boost::replace_last (obj_indices_fn, ".pcd", ".txt");
                std::ifstream f ( obj_indices_fn.c_str() );
                int idx;
                while (f >> idx)
                    model.indices_[i].indices.push_back(idx);
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

    /**
         * \brief Creates the model representation of the training set, generating views if needed
         */
    void
    generate ()
    {
        models_.clear();
        std::vector < std::string > model_files = io::getFilesInDirectory (path_, ".3D_model.pcd",  true);
        std::cout << "There are " << model_files.size() << " models." << std::endl;

        for (size_t i = 0; i < model_files.size (); i++)
        {
            ModelTPtr m(new ModelT);

            std::vector < std::string > strs;
            boost::split (strs, model_files[i], boost::is_any_of ("/\\"));
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
            loadModel (*m);
            models_.push_back (m);
        }
        this->createVoxelGridAndDistanceTransform(resolution_);
    }
};
}

#endif
