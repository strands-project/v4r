#include <v4r/common/miscellaneous.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/model.h>

#include <sstream>

#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/pcd_io.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace v4r
{
template<typename PointT>
void
Model<PointT>::computeNormalsAssembledCloud(float radius_normals)
{
    typename pcl::search::KdTree<PointT>::Ptr normals_tree (new pcl::search::KdTree<PointT>);
    typedef typename pcl::NormalEstimationOMP<PointT, pcl::Normal> NormalEstimator_;
    NormalEstimator_ n3d;
    normals_assembled_.reset (new pcl::PointCloud<pcl::Normal> ());
    normals_tree->setInputCloud (assembled_);
    n3d.setRadiusSearch (radius_normals);
    n3d.setSearchMethod (normals_tree);
    n3d.setInputCloud (assembled_);
    n3d.compute (*normals_assembled_);
}


template<typename PointT>
typename pcl::PointCloud<PointT>::ConstPtr
Model<PointT>::getAssembled (int resolution_mm) const
{
    if(resolution_mm <= 0)
        return assembled_;

    const auto it = voxelized_assembled_.find (resolution_mm);
    if (it == voxelized_assembled_.end ())
    {
        double resolution = (double)resolution_mm / 1000.;
        PointTPtr voxelized (new pcl::PointCloud<PointT>);
        pcl::VoxelGrid<PointT> grid;
        grid.setInputCloud (assembled_);
        grid.setLeafSize (resolution, resolution, resolution);
        grid.setDownsampleAllData(true);
        grid.filter (*voxelized);

        PointTPtrConst voxelized_const (new pcl::PointCloud<PointT> (*voxelized));
        voxelized_assembled_[resolution_mm] = voxelized_const;
        return voxelized_const;
    }

    return it->second;
}

template<typename PointT>
pcl::PointCloud<pcl::Normal>::ConstPtr
Model<PointT>::getNormalsAssembled (int resolution_mm) const
{
    if(resolution_mm <= 0)
        return normals_assembled_;


    const auto it = normals_voxelized_assembled_.find (resolution_mm);
    if (it == normals_voxelized_assembled_.end ())
    {
        double resolution = resolution_mm / 1000.f;
        pcl::PointCloud<pcl::PointNormal>::Ptr voxelized (new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr assembled_with_normals (new pcl::PointCloud<pcl::PointNormal>);
        assembled_with_normals->points.resize(assembled_->points.size());
        assembled_with_normals->width = assembled_->width;
        assembled_with_normals->height = assembled_->height;

        for(size_t i=0; i < assembled_->points.size(); i++)
        {
            assembled_with_normals->points[i].getVector4fMap() = assembled_->points[i].getVector4fMap();
            assembled_with_normals->points[i].getNormalVector4fMap() = normals_assembled_->points[i].getNormalVector4fMap();
        }

        pcl::VoxelGrid<pcl::PointNormal> grid;
        grid.setInputCloud (assembled_with_normals);
        grid.setLeafSize (resolution, resolution, resolution);
        grid.setDownsampleAllData(true);
        grid.filter (*voxelized);

        pcl::PointCloud<pcl::Normal>::Ptr voxelized_const (new pcl::PointCloud<pcl::Normal> ());
        voxelized_const->points.resize(voxelized->points.size());
        voxelized_const->width = voxelized->width;
        voxelized_const->height = voxelized->height;

        for(size_t i=0; i < voxelized_const->points.size(); i++)
            voxelized_const->points[i].getNormalVector4fMap() = voxelized->points[i].getNormalVector4fMap();


        normals_voxelized_assembled_[resolution_mm] = voxelized_const;
        return voxelized_const;
    }

    return it->second;
}

template<typename PointT>
pcl::PointCloud<pcl::PointXYZL>::Ptr
Model<PointT>::getAssembledSmoothFaces (int resolution_mm)
{
    if(resolution_mm <= 0)
        return faces_cloud_labels_;

    const auto it = voxelized_assembled_labels_.find (resolution_mm);
    if (it == voxelized_assembled_labels_.end ())
    {
        double resolution = resolution_mm / (double)1000.f;
        pcl::PointCloud<pcl::PointXYZL>::Ptr voxelized (new pcl::PointCloud<pcl::PointXYZL>);
        pcl::VoxelGrid<pcl::PointXYZL> grid;
        grid.setInputCloud (faces_cloud_labels_);
        grid.setLeafSize (resolution, resolution, resolution);
        grid.setDownsampleAllData(true);
        grid.filter (*voxelized);

        voxelized_assembled_labels_[resolution_mm] = voxelized;
        return voxelized;
    }

    return it->second;
}

template<typename PointT>
void
Model<PointT>::initialize(const std::string &model_filename)
{
    typename pcl::PointCloud<PointTWithNormal>::Ptr all_assembled (new pcl::PointCloud<PointTWithNormal>);
    if ( !io::existsFile(model_filename) || pcl::io::loadPCDFile(model_filename, *all_assembled) == -1 )
    {
        pcl::ScopeTime t("Creating 3D model");
        typename pcl::PointCloud<PointT>::Ptr accumulated_cloud (new pcl::PointCloud<PointT>); ///< 3D point cloud taking into account all training views
        pcl::PointCloud<pcl::Normal>::Ptr accumulated_normals (new pcl::PointCloud<pcl::Normal>); /// corresponding normals to the 3D point cloud taking into account all training views

        ///TODO use noise model and voxel grid to merge point clouds. For now just accumulate all points.
        for(const typename TrainingView<PointT>::ConstPtr &v : views_)
        {
            typename pcl::PointCloud<PointT>::ConstPtr cloud;
            pcl::PointCloud<pcl::Normal>::ConstPtr normals;
            std::vector<int> indices;
            Eigen::Matrix4f tf;
            typename pcl::PointCloud<PointT>::Ptr obj_cloud_tmp (new pcl::PointCloud<PointT>);
            typename pcl::PointCloud<pcl::Normal>::Ptr obj_normals_tmp (new pcl::PointCloud<pcl::Normal>);

            if(v->cloud_)
            {
                cloud = v->cloud_;
                indices = v->indices_;
                pcl::copyPointCloud( *v->cloud_, indices, *obj_cloud_tmp );
            }
            else
            {
                typename pcl::PointCloud<PointT>::Ptr cloud_tmp (new pcl::PointCloud<PointT>);
                pcl::io::loadPCDFile(v->filename_, *cloud_tmp);
                cloud = cloud_tmp;
            }


            if(v->normals_)
                normals = v->normals_;
            else
            {
                pcl::PointCloud<pcl::Normal>::Ptr normals_tmp(new pcl::PointCloud<pcl::Normal>);

                if( cloud->isOrganized() )
                {
                    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
                    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
                    ne.setMaxDepthChangeFactor(0.02f);
                    ne.setNormalSmoothingSize(10.0f);
                    ne.setInputCloud(cloud);
                    ne.compute(*normals_tmp);
                }
                else
                {
                    typename pcl::search::KdTree<PointT>::Ptr normals_tree (new pcl::search::KdTree<PointT>);
                    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
                    normals_tree->setInputCloud (cloud);
                    ne.setRadiusSearch (0.02f);
                    ne.setSearchMethod (normals_tree);
                    ne.setInputCloud (cloud);
                    ne.compute (*normals_tmp);
                }
                normals = normals_tmp;
            }


            if( !v->indices_.empty() )
                indices = v->indices_;
            else
            {
                std::ifstream mi_f ( v->indices_filename_ );
                int idx;
                while ( mi_f >> idx )
                    indices.push_back(idx);
                mi_f.close();
            }


            try
            {
                tf = io::readMatrixFromFile(v->pose_filename_);
            }
            catch(const std::runtime_error &e)
            {
                tf = Eigen::Matrix4f::Identity();
            }

            if( indices.empty() )
            {
                pcl::copyPointCloud(*cloud, *obj_cloud_tmp);
                pcl::copyPointCloud( *normals, *obj_normals_tmp );
            }
            else
            {
                pcl::copyPointCloud(*cloud, indices, *obj_cloud_tmp);
                pcl::copyPointCloud( *normals, indices, *obj_normals_tmp );
            }
            pcl::transformPointCloud( *obj_cloud_tmp, *obj_cloud_tmp, tf);
            transformNormals(*obj_normals_tmp, *obj_normals_tmp, tf);

            *accumulated_cloud += *obj_cloud_tmp;
            *accumulated_normals += *obj_normals_tmp;
        }

        if(accumulated_cloud->points.size() != accumulated_normals->points.size())
            std::cerr << "Point cloud and normals point cloud of model created by accumulating all points from training does not have the same size! This can lead to undefined behaviour!" << std::endl;

        all_assembled.reset (new pcl::PointCloud<PointTWithNormal>);
        pcl::concatenateFields (*accumulated_cloud, *accumulated_normals, *all_assembled);

        if( !model_filename.empty() )
        {
            io::createDirForFileIfNotExist ( model_filename );
            pcl::io::savePCDFileBinaryCompressed ( model_filename, *all_assembled);
        }
    }
    else
        pcl::io::loadPCDFile(model_filename, *all_assembled);

    assembled_.reset(new pcl::PointCloud<PointT>);
    normals_assembled_.reset(new pcl::PointCloud<pcl::Normal>);
    pcl::copyPointCloud( * all_assembled, *assembled_);
    pcl::copyPointCloud( * all_assembled, *normals_assembled_);
    pcl::getMinMax3D(*assembled_, minPoint_, maxPoint_);
    pcl::compute3DCentroid(*assembled_, centroid_);
}

//template<typename PointT>
//template<class Archive>
//void
//Model<PointT>::serialize(Archive & ar, const unsigned int version)
//{
//    ar & indices_;
//    ar & poses_;
//    ar & eigen_pose_alignment_;
//    ar & elongations_;
//    ar & self_occlusions_;
//    ar & class_;
//    ar & id_;
//    ar & normals_assembled_;
//    ar & view_filenames_;
//    ar & *keypoints_;
//    ar & *kp_normals_;
//    ar & centroid_;
//    ar & view_centroid_;
//    ar & centroid_computed_;
//    ar & flip_normals_based_on_vp_;
//}

template class V4R_EXPORTS Model<pcl::PointXYZ>;
template class V4R_EXPORTS Model<pcl::PointXYZRGB>;

//template V4R_EXPORTS void Model<pcl::PointXYZRGB>::serialize<boost::archive>(boost::archive & ar, const unsigned int version);

}

