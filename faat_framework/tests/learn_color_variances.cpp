/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/model_only_source.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <pcl/features/organized_edge_detection.h>
#include <faat_pcl/utils/miscellaneous.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <faat_pcl/utils/noise_models.h>
#include <faat_pcl/registration/visibility_reasoning.h>
#include <pcl/common/angles.h>
#include "pcl/registration/icp.h"
#include "faat_pcl/recognition/hv/occlusion_reasoning.h"
#include "faat_pcl/recognition/impl/hv/occlusion_reasoning.hpp"

float Z_DIST_ = 1.5f;
std::string GT_DIR_;
std::string MODELS_DIR_;
std::string MODELS_DIR_FOR_VIS_;
float model_scale = 1.f;
int SCENE_STEP_ = 1;
bool specify_color_ = true;

#define VISUALIZE_
//#define VISUALIZE_SINGLE_

void
getModelsInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext)
{
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
    {
        //check if its a directory, then get models in it
        if (bf::is_directory (*itr))
        {
#if BOOST_FILESYSTEM_VERSION == 3
            std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#else
            std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

            bf::path curr_path = itr->path ();
            getModelsInDirectory (curr_path, so_far, relative_paths, ext);
        }
        else
        {
            //check that it is a ply file and then add, otherwise ignore..
            std::vector<std::string> strs;
#if BOOST_FILESYSTEM_VERSION == 3
            std::string file = (itr->path ().filename ()).string ();
#else
            std::string file = (itr->path ()).filename ();
#endif

            boost::split (strs, file, boost::is_any_of ("."));
            std::string extension = strs[strs.size () - 1];

            if (extension.compare (ext) == 0)
            {
#if BOOST_FILESYSTEM_VERSION == 3
                std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
                std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

                relative_paths.push_back (path);
            }
        }
    }
}

static float sRGB_LUT[256] = {- 1};
static float sXYZ_LUT[4000] = {- 1};

void
RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
{
    if (sRGB_LUT[0] < 0)
    {
        for (int i = 0; i < 256; i++)
        {
            float f = static_cast<float> (i) / 255.0f;
            if (f > 0.04045)
                sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
            else
                sRGB_LUT[i] = f / 12.92f;
        }

        for (int i = 0; i < 4000; i++)
        {
            float f = static_cast<float> (i) / 4000.0f;
            if (f > 0.008856)
                sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
            else
                sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
        }
    }

    float fr = sRGB_LUT[R];
    float fg = sRGB_LUT[G];
    float fb = sRGB_LUT[B];

    // Use white = D65
    const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
    const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
    const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

    float vx = x / 0.95047f;
    float vy = y;
    float vz = z / 1.08883f;

    vx = sXYZ_LUT[int(vx*4000)];
    vy = sXYZ_LUT[int(vy*4000)];
    vz = sXYZ_LUT[int(vz*4000)];

    L = 116.0f * vy - 16.0f;
    if (L > 100)
        L = 100.0f;

    A = 500.0f * (vx - vy);
    if (A > 120)
        A = 120.0f;
    else if (A <- 120)
        A = -120.0f;

    B2 = 200.0f * (vy - vz);
    if (B2 > 120)
        B2 = 120.0f;
    else if (B2<- 120)
        B2 = -120.0f;
}

std::vector<Eigen::Vector3f> scene_LAB_values_;
std::map<std::string, pcl::PointCloud<pcl::PointXYZL>::Ptr> same_colors_;
std::map<std::string, std::vector<std::vector<int> > > grouped_by_color_;

template<typename PointT>
void getLABValue(Eigen::Vector3f & lab,
                 typename pcl::PointCloud<PointT>::Ptr & cloud,
                 int ii)
{
    bool exists_s;
    float rgb_s;

    typedef pcl::PointCloud<PointT> CloudS;
    typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;

    pcl::for_each_type<FieldListS> (
                pcl::CopyIfFieldExists<typename CloudS::PointType, float> (cloud->points[ii],"rgb", exists_s, rgb_s)
                );

    if (exists_s)
    {

        uint32_t rgb = *reinterpret_cast<int*> (&rgb_s);
        unsigned char rs = (rgb >> 16) & 0x0000ff;
        unsigned char gs = (rgb >> 8) & 0x0000ff;
        unsigned char bs = (rgb) & 0x0000ff;

        float LRefs, aRefs, bRefs;

        RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
        LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

        lab = (Eigen::Vector3f(LRefs, aRefs, bRefs));
    }
}

void
computeGSHistogram
(std::vector<float> & gs_values, Eigen::MatrixXf & histogram, int hist_size)
{
    float max = 255.f;
    float min = 0.f;
    int dim = 1;

    histogram = Eigen::MatrixXf (hist_size, dim);
    histogram.setZero ();
    for (size_t j = 0; j < gs_values.size (); j++)
    {
        int pos = std::floor (static_cast<float> (gs_values[j] - min) / (max - min) * hist_size);
        if(pos < 0)
            pos = 0;

        if(pos > hist_size)
            pos = hist_size - 1;

        histogram (pos, 0)++;
    }
}

void
specifyRGBHistograms (Eigen::MatrixXf & src, Eigen::MatrixXf & dst, Eigen::MatrixXf & lookup, int dim)
{
    //normalize histograms
    for(size_t i=0; i < dim; i++) {
        src.col(i) /= src.col(i).sum();
        dst.col(i) /= dst.col(i).sum();
    }

    Eigen::MatrixXf src_cumulative(src.rows(), dim);
    Eigen::MatrixXf dst_cumulative(dst.rows(), dim);
    lookup = Eigen::MatrixXf(src.rows(), dim);
    lookup.setZero();

    src_cumulative.setZero();
    dst_cumulative.setZero();

    for (size_t i = 0; i < dim; i++)
    {
        src_cumulative (0, i) = src (0, i);
        dst_cumulative (0, i) = dst (0, i);
        for (size_t j = 1; j < src_cumulative.rows (); j++)
        {
            src_cumulative (j, i) = src_cumulative (j - 1, i) + src (j, i);
            dst_cumulative (j, i) = dst_cumulative (j - 1, i) + dst (j, i);
        }

        int last = 0;
        for (int k = 0; k < src_cumulative.rows (); k++)
        {
            for (int z = last; z < src_cumulative.rows (); z++)
            {
                if (src_cumulative (z, i) - dst_cumulative (k, i) >= 0)
                {
                    if (z > 0 && (dst_cumulative (k, i) - src_cumulative (z - 1, i)) < (src_cumulative (z, i) - dst_cumulative (k, i)))
                        z--;

                    lookup (k, i) = z;
                    last = z;
                    break;
                }
            }
        }

        int min = 0;
        for (int k = 0; k < src_cumulative.rows (); k++)
        {
            if (lookup (k, i) != 0)
            {
                min = lookup (k, i);
                break;
            }
        }

        for (int k = 0; k < src_cumulative.rows (); k++)
        {
            if (lookup (k, i) == 0)
                lookup (k, i) = min;
            else
                break;
        }

        //max mapping extension
        int max = 0;
        for (int k = (src_cumulative.rows () - 1); k >= 0; k--)
        {
            if (lookup (k, i) != 0)
            {
                max = lookup (k, i);
                break;
            }
        }

        for (int k = (src_cumulative.rows () - 1); k >= 0; k--)
        {
            if (lookup (k, i) == 0)
                lookup (k, i) = max;
            else
                break;
        }
    }
}

template<typename PointT>
void
specifyColor(typename boost::shared_ptr<pcl::octree::OctreePointCloudSearch<PointT> > octree_scene_downsampled_,
             typename pcl::PointCloud<PointT>::Ptr & recog_model_cloud,
             pcl::PointCloud<pcl::PointXYZL>::Ptr & smooth_faces,
             std::vector<int> & visible_indices,
             std::vector<Eigen::Vector3f> & model_LAB)
{

    typename pcl::PointCloud<PointT>::ConstPtr scene_cloud = octree_scene_downsampled_->getInputCloud();

    typename pcl::PointCloud<PointT>::Ptr model_cloud_specified(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*recog_model_cloud, *model_cloud_specified);

    std::vector< std::vector<int> > label_indices;
    std::vector< std::set<int> > label_explained_indices_points;

    pcl::PointCloud<pcl::PointXYZL>::Ptr visible_labels(new pcl::PointCloud<pcl::PointXYZL>);

    //use visible indices to check which points are visible
    pcl::copyPointCloud(*smooth_faces, visible_indices, *visible_labels);

    //specify using the smooth faces
    int max_label = 0;
    for(size_t k=0; k < visible_labels->points.size(); k++)
    {
        if(visible_labels->points[k].label > max_label)
        {
            max_label = visible_labels->points[k].label;
        }
    }

    //1) group points based on label
    label_indices.resize(max_label + 1);
    for(size_t k=0; k < visible_labels->points.size(); k++)
    {
        label_indices[visible_labels->points[k].label].push_back(k);
    }

    //2) for each group, find corresponding scene points and push them into label_explained_indices_points
    std::vector<std::pair<int, float> > label_index_distances;
    label_index_distances.resize(scene_cloud->points.size(), std::make_pair(-1, std::numeric_limits<float>::infinity()));

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;

    for(size_t j=0; j < label_indices.size(); j++)
    {
        for (size_t i = 0; i < label_indices[j].size (); i++)
        {
            if (octree_scene_downsampled_->radiusSearch (recog_model_cloud->points[label_indices[j][i]], 0.005f,
                                                         nn_indices, nn_distances, std::numeric_limits<int>::max ()) > 0)
            {
                for (size_t k = 0; k < nn_distances.size (); k++)
                {
                    if(label_index_distances[nn_indices[k]].first == static_cast<int>(j))
                    {
                        //already explained by the same label
                    }
                    else
                    {
                        //if different labels, then take the new label if distances is smaller
                        if(nn_distances[k] < label_index_distances[nn_indices[k]].second)
                        {
                            label_index_distances[nn_indices[k]].first = static_cast<int>(j);
                            label_index_distances[nn_indices[k]].second = nn_distances[k];
                        } //otherwise, ignore new label since the older one is closer
                    }
                }
            }
        }
    }

    //3) set label_explained_indices_points
    label_explained_indices_points.resize(max_label + 1);
    for (size_t i = 0; i < scene_cloud->points.size (); i++)
    {
        if(label_index_distances[i].first < 0)
            continue;

        label_explained_indices_points[label_index_distances[i].first].insert(i);
    }

    //specify each label
    for(size_t j=0; j < label_indices.size(); j++)
    {
        std::set<int> explained_indices_points = label_explained_indices_points[j];

        std::vector<float> model_gs_values, scene_gs_values;

        //compute RGB histogram for the model points
        for (size_t i = 0; i < label_indices[j].size (); i++)
        {
            model_gs_values.push_back(model_LAB[label_indices[j][i]][0] * 255.f);
        }

        //compute RGB histogram for the explained points
        std::set<int>::iterator it;
        for(it=explained_indices_points.begin(); it != explained_indices_points.end(); it++)
        {
            scene_gs_values.push_back(scene_LAB_values_[*it][0] * 255.f);
        }

        Eigen::MatrixXf gs_model, gs_scene;
        computeGSHistogram(model_gs_values, gs_model, 100);
        computeGSHistogram(scene_gs_values, gs_scene, 100);
        int hist_size = gs_model.rows();

        //histogram specification, adapt model values to scene values
        Eigen::MatrixXf lookup;
        specifyRGBHistograms(gs_scene, gs_model, lookup, 1);

        for (size_t i = 0; i < label_indices[j].size (); i++)
        {
            float LRefm = model_LAB[label_indices[j][i]][0] * 255.f;
            int pos = std::floor (static_cast<float> (LRefm) / 255.f * hist_size);
            assert(pos < lookup.rows());
            float gs_specified = lookup(pos, 0);
            //std::cout << "gs specified:" << gs_specified << " size:" << hist_size << std::endl;
            LRefm = gs_specified * (255.f / static_cast<float>(hist_size)) / 255.f;
            model_LAB[label_indices[j][i]][0] = LRefm;
            //recog_model->cloud_indices_specified_.push_back(label_indices[j][i]);
        }
    }
}

template<typename PointT>
void
groupSimilarColors(typename pcl::PointCloud<PointT>::Ptr & model_cloud,
                   std::vector<Eigen::Vector3f> & model_LAB,
                   pcl::PointCloud<pcl::PointXYZL>::Ptr & same_colors,
                   std::vector<std::vector<int> > & grouped_by_color,
                   std::vector<int> & point_index_to_bucket,
                   std::vector<Eigen::Vector3f> & bucket_average)
{
    //ranges are 0,1 for L and -1,1 for A,B
    float L_step, AB_step;
    L_step = 0.1;
    AB_step = 0.1;

    int size_L = static_cast<int>(1 / L_step);
    int size_AB = static_cast<int>(2 / AB_step);
    int size = size_L * size_AB * size_AB;

    std::cout << "bucket size:" << size << std::endl;
    grouped_by_color.resize(size);
    point_index_to_bucket.resize(model_cloud->points.size());
    for(size_t i=0; i < model_cloud->points.size(); i++)
    {
        Eigen::Vector3f lab = model_LAB[i];
        //quantize lab
        int idx_L, idx_A, idx_B;
        idx_L = std::floor(lab[0] / L_step);
        assert(idx_L < size_L && idx_L >= 0);
        idx_A = std::floor( (lab[1] + 1.f) / AB_step);
        assert(idx_A < size_AB && idx_A >= 0);
        idx_B = std::floor( (lab[2] + 1.f) / AB_step);
        assert(idx_B < size_AB && idx_B >= 0);
        int idx = idx_L * (size_AB * size_AB) + idx_A * size_AB + idx_B;
        assert(idx >= 0 && idx < size);
        grouped_by_color[idx].push_back(static_cast<int>(i));
        point_index_to_bucket[i] = idx;
    }

    same_colors.reset(new pcl::PointCloud<pcl::PointXYZL>);
    same_colors->points.resize(model_cloud->points.size());
    int label=0;
    bucket_average.resize(grouped_by_color.size());
    int max_size = 0;
    for(size_t i=0; i < grouped_by_color.size(); i++)
    {
        if(grouped_by_color[i].size() > max_size)
            max_size = grouped_by_color[i].size();

        if(grouped_by_color[i].size() > 0)
        {
            Eigen::Vector3f average_color(0,0,0);
            for(size_t k=0; k < grouped_by_color[i].size(); k++)
            {
                same_colors->points[grouped_by_color[i][k]].getVector3fMap() = model_cloud->points[grouped_by_color[i][k]].getVector3fMap();
                same_colors->points[grouped_by_color[i][k]].label = label;
                for(size_t j=0; j < 3; j++)
                {
                    average_color[j] += model_LAB[grouped_by_color[i][k]][j];
                }
            }

            for(size_t j=0; j < 3; j++)
            {
                average_color[j] /= static_cast<float>(grouped_by_color[i].size());
            }

            bucket_average[i] = average_color;
            //std::cout << average_color << std::endl;
        }
        else
        {
            bucket_average[i] = Eigen::Vector3f(-10,-10,-10);
        }

        label++;
    }

    std::cout << "max size:" << max_size << std::endl;

    int max_label = label;
    std::vector<uint32_t> label_colors;
    label_colors.reserve (max_label);
    srand (static_cast<unsigned int> (time (0)));
    while (label_colors.size () <= max_label )
    {
        uint8_t r = static_cast<uint8_t>( (rand () % 256));
        uint8_t g = static_cast<uint8_t>( (rand () % 256));
        uint8_t b = static_cast<uint8_t>( (rand () % 256));
        label_colors.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_labels;
    rgb_labels.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    rgb_labels->points.resize(same_colors->points.size());
    for(size_t i=0; i < same_colors->points.size(); i++)
    {
        rgb_labels->points[i].getVector3fMap() = same_colors->points[i].getVector3fMap();
        uint32_t rgb = *reinterpret_cast<int*> (&label_colors[same_colors->points[i].label]);
        rgb_labels->points[i].rgb = rgb;
    }

    rgb_labels->width = same_colors->points.size();
    rgb_labels->height = 1;

    /*pcl::visualization::PCLVisualizer vis("grouped colors");
    int v1,v2;
    vis.createViewPort(0,0,0.5,1,v1);
    vis.createViewPort(0.5,0,1,1,v2);
    vis.addPointCloud(model_cloud, "model", v1);

//    for(size_t i=0; i < grouped_by_color.size(); i++)
//    {

//        if(grouped_by_color[i].size() > 50)
//        {
//            pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_labels_same(new pcl::PointCloud<pcl::PointXYZRGB>);
//            pcl::copyPointCloud(*model_cloud, grouped_by_color[i], *rgb_labels_same);

//            vis.addPointCloud(rgb_labels_same, "grouped_colors", v2);
//            vis.spin();
//            vis.removePointCloud("grouped_colors", v2);
//        }
//    }

    vis.addPointCloud(rgb_labels, "grouped_colors", v2);

    //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler(same_colors, "label");
    //vis.addPointCloud<pcl::PointXYZL>(same_colors, handler, "grouped_colors", v2);
    vis.spin();*/
}

template<typename PointT>
void
learnColorVariance(typename pcl::PointCloud<PointT>::Ptr & model_cloud,
                   typename pcl::PointCloud<PointT>::Ptr & scene_cloud,
                   pcl::PointCloud<pcl::PointXYZL>::Ptr & faces_aligned,
                   pcl::PointCloud<pcl::PointXYZI>::Ptr & variance_L,
                   pcl::PointCloud<pcl::PointXYZI>::Ptr & variance_AB,
                   std::string & id,
                   float threshold=0.01f,
                   bool do_icp=true)
{

#ifdef VISUALIZE_SINGLE_
    pcl::visualization::PCLVisualizer vis ("Recognition results");
    int v1, v2, v3, v4;
    vis.createViewPort (0.0, 0.0, 0.25, 1.0, v1);
    vis.createViewPort (0.25, 0, 0.5, 1.0, v2);
    vis.createViewPort (0.5, 0, 0.75, 1.0, v3);
    vis.createViewPort (0.75, 0, 1, 1.0, v4);

    vis.addPointCloud(scene_cloud, "scene", v1);
    //vis.addPointCloud(model_cloud, "model", v2);
#endif

    //for each model point, get the neighbours in the scene
    typename boost::shared_ptr<pcl::octree::OctreePointCloudSearch<PointT> > octree_scene_downsampled_;
    octree_scene_downsampled_.reset(new pcl::octree::OctreePointCloudSearch<PointT>(0.01f));
    octree_scene_downsampled_->setInputCloud(scene_cloud);
    octree_scene_downsampled_->addPointsFromInputCloud();

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    std::set<int> under_the_influence;

    if(do_icp)
    {
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(model_cloud);
        icp.setInputTarget(scene_cloud);
        icp.setMaxCorrespondenceDistance(0.005f);
        icp.setMaximumIterations(10);
        icp.setEuclideanFitnessEpsilon(1e-12);

        typename pcl::PointCloud<PointT>::Ptr output (new pcl::PointCloud<PointT> ());
        icp.align(*output);
        Eigen::Matrix4f final_trans;
        final_trans = icp.getFinalTransformation();
        pcl::transformPointCloud(*model_cloud, *model_cloud, final_trans);
    }

    std::vector<Eigen::Vector3f> model_LAB;
    for (size_t i = 0; i < model_cloud->points.size (); i++)
    {
        Eigen::Vector3f lab_model;
        getLABValue<PointT>(lab_model, model_cloud, i);
        model_LAB.push_back(lab_model);
    }

    pcl::PointCloud<pcl::PointXYZL>::Ptr same_colors;
    std::vector<std::vector<int> > grouped_by_color;
    std::vector<int> point_index_to_bucket;
    std::vector<Eigen::Vector3f> bucket_average;

    groupSimilarColors<PointT>(model_cloud, model_LAB, same_colors, grouped_by_color, point_index_to_bucket, bucket_average);

    std::vector<std::vector<Eigen::Vector3f> > bucket_min_differences;
    bucket_min_differences.resize(grouped_by_color.size());

    //compute visibility indices
    std::vector<int> visible_indices;

    typename pcl::PointCloud<PointT>::Ptr filtered (new pcl::PointCloud<PointT> ());
    typename pcl::PointCloud<PointT>::ConstPtr const_filtered(new pcl::PointCloud<PointT> (*model_cloud));
    typename pcl::PointCloud<PointT>::ConstPtr scene_const(new pcl::PointCloud<PointT> (*scene_cloud));

    filtered = faat_pcl::occlusion_reasoning::filter<PointT,PointT> (scene_const, const_filtered, 525.f, 0.01, visible_indices);
    //pcl::copyPointCloud(*const_filtered, visible_indices, *model_cloud);

    /*if(specify_color_)
        specifyColor(octree_scene_downsampled_, model_cloud, faces_aligned, visible_indices, model_LAB);*/

    for (size_t i = 0; i < visible_indices.size (); i++)
    {

        int idx_bucket = point_index_to_bucket[visible_indices[i]];

        Eigen::Vector3f lab_model;
        lab_model = model_LAB[visible_indices[i]];

        if (octree_scene_downsampled_->radiusSearch (model_cloud->points[visible_indices[i]], threshold,
                                                     nn_indices, nn_distances, std::numeric_limits<int>::max ()) > 0)
        {

            /*if(nn_indices.size() < 2)
                continue;*/

            Eigen::Vector3f min_diff(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            Eigen::Vector3f avg_diff(0,0,0);

            float min_diff_euc = std::numeric_limits<float>::max();
            int idx = 0;
            for(size_t k=0; k < nn_indices.size(); k++)
            {
                under_the_influence.insert(nn_indices[k]);
                float min_diff_euc_point = 0;

                for(size_t j=0; j < 3; j++)
                {
                    float diff = lab_model[j] - scene_LAB_values_[nn_indices[k]][j];
                    min_diff_euc_point += diff * diff;
                    avg_diff[j] += std::abs(diff);
                }

                min_diff_euc_point = sqrt(min_diff_euc_point);
                if(min_diff_euc_point < min_diff_euc)
                {
                    min_diff_euc = min_diff_euc_point;
                    idx = k;
                }
            }

            for(size_t j=0; j < 3; j++)
            {
                float diff = lab_model[j] - scene_LAB_values_[nn_indices[idx]][j];
                min_diff[j] = std::abs(diff);
                avg_diff[j] /= static_cast<float>(nn_indices.size());
            }

            bucket_min_differences[idx_bucket].push_back(min_diff);
            //bucket_min_differences[idx_bucket].push_back(avg_diff);
        }
    }

    std::vector<Eigen::Vector3f> bucket_average_difference;
    bucket_average_difference.resize(grouped_by_color.size(), Eigen::Vector3f(0,0,0));

    //for each bucket, compute mean
    for(size_t i=0; i < bucket_min_differences.size(); i++)
    {
        for(size_t k=0; k < bucket_min_differences[i].size(); k++)
        {
            for(size_t j=0; j < 3; j++)
            {
                bucket_average_difference[i][j] += bucket_min_differences[i][k][j];
            }
        }

        for(size_t j=0; j < 3; j++)
        {
            bucket_average_difference[i][j] /= static_cast<float>(bucket_min_differences[i].size());
        }
    }

    std::vector<Eigen::Vector3f> bucket_sigma;
    bucket_sigma.resize(grouped_by_color.size(), Eigen::Vector3f(0,0,0));

    //for each bucket, compute variance
    for(size_t i=0; i < bucket_min_differences.size(); i++)
    {
        for(size_t k=0; k < bucket_min_differences[i].size(); k++)
        {
            for(size_t j=0; j < 3; j++)
            {
                float diff = bucket_min_differences[i][k][j] - bucket_average_difference[i][j];
                bucket_sigma[i][j] += diff * diff;
            }
        }

        for(size_t j=0; j < 3; j++)
        {
            bucket_sigma[i][j] /= static_cast<float>(bucket_min_differences[i].size() - 1);
            bucket_sigma[i][j] = bucket_average_difference[i][j] + sqrt(bucket_sigma[i][j]);
        }
    }

    variance_L->points.resize(model_cloud->points.size());
    variance_AB->points.resize(model_cloud->points.size());

    int num = 0;

    for (size_t i = 0; i < model_cloud->points.size (); i++)
    //for (size_t k = 0; k < visible_indices.size (); k++)
    {
        //int i = visible_indices[k];
        int idx_bucket = point_index_to_bucket[i];
        if(bucket_min_differences[idx_bucket].size() > 0)
        {
            variance_L->points[num].getVector3fMap() = model_cloud->points[i].getVector3fMap();
            variance_AB->points[num].getVector3fMap() = model_cloud->points[i].getVector3fMap();

            variance_L->points[num].intensity = bucket_sigma[idx_bucket][0];
            variance_AB->points[num].intensity = (bucket_sigma[idx_bucket][1] + bucket_sigma[idx_bucket][2]) / 2.f;
            num++;
        }
    }

    variance_L->points.resize(num);
    variance_AB->points.resize(num);
    variance_L->width = variance_AB->width = num;
    variance_L->height = variance_AB->height = 1;
/*#ifdef VISUALIZE_SINGLE_
    std::vector<int> output(under_the_influence.begin(), under_the_influence.end());
    typename pcl::PointCloud<PointT>::Ptr scene_influence(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*scene_cloud, output, *scene_influence);

    {
        for(size_t k=0; k < output.size(); k++)
        {
            scene_influence->points[k].r = scene_influence->points[k].g = scene_influence->points[k].b = scene_LAB_values_[output[k]][0] * 255;
            //scene_influence->points[k].g = (scene_LAB_values_[output[k]][1] + 1.f) / 2.f * 255;
            //scene_influence->points[k].b = (scene_LAB_values_[output[k]][2] + 1.f) / 2.f * 255;
        }

        vis.addPointCloud(scene_influence, "scene_influence", v3);
    }

    vis.spin();
#endif*/

}

template<typename PointTModel, typename PointT>
void
recognizeAndVisualize (typename boost::shared_ptr<faat_pcl::rec_3d_framework::ModelOnlySource<PointTModel, PointT> > & source,
                       std::string & scene_file,
                       std::string model_ext = "ply")
{

    faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;
    or_eval.setGTDir(GT_DIR_);
    or_eval.setModelsDir(MODELS_DIR_);
    or_eval.setCheckPose(true);
    or_eval.setScenesDir(scene_file);
    or_eval.setModelFileExtension(model_ext);
    or_eval.setReplaceModelExtension(false);
    or_eval.setDataSource(source);
    or_eval.setCheckPose(true);
    or_eval.useMaxOcclusion(true);

    typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;

    typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    bf::path input = scene_file;
    std::vector<std::string> files_to_recognize;

    if (bf::is_directory (input))
    {
        std::vector<std::string> files;
        std::string start = "";
        std::string ext = std::string ("pcd");
        bf::path dir = input;
        getModelsInDirectory (dir, start, files, ext);
        std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
        for (size_t i = 0; i < files.size (); i++)
        {
            std::cout << files[i] << std::endl;
            std::stringstream filestr;
            filestr << scene_file << files[i];
            std::string file = filestr.str ();
            files_to_recognize.push_back (file);
        }

        std::sort (files_to_recognize.begin (), files_to_recognize.end ());

        if(SCENE_STEP_ > 1)
        {
            std::map<std::string, bool> ignore_list;

            //some scenes will not be recognized, modify files to recognize accordingly
            std::vector<std::string> files_to_recognize_step;
            for(size_t i=0; i < files_to_recognize.size(); i++)
            {
                if( ((int)(i) % SCENE_STEP_) == 0)
                {
                    files_to_recognize_step.push_back(files_to_recognize[i]);
                }
                else
                {
                    std::string file_to_recognize = files_to_recognize[i];
                    boost::replace_all (file_to_recognize, scene_file, "");
                    ignore_list.insert(std::make_pair(file_to_recognize, true));
                }
            }

            std::cout << files_to_recognize.size() << " " << files_to_recognize_step.size() << std::endl;
            files_to_recognize = files_to_recognize_step;
            or_eval.setIgnoreList(ignore_list);
        }

        or_eval.setScenesDir(scene_file);
        or_eval.loadGTData();
    }
    else
    {
        files_to_recognize.push_back (scene_file);
    }

    boost::shared_ptr<std::vector<ModelTPtr> > all_models = source->getModels();
    for(size_t m=0; m < all_models->size(); m++)
    {
        std::cout << all_models->at(m)->id_ << std::endl;
        ConstPointInTPtr model_cloud_const = all_models->at(m)->getAssembled (-1);
        PointInTPtr model_cloud(new pcl::PointCloud<PointT>(*model_cloud_const));

        std::vector<Eigen::Vector3f> model_LAB;
        for (size_t i = 0; i < model_cloud->points.size (); i++)
        {
            Eigen::Vector3f lab_model;
            getLABValue<PointT>(lab_model, model_cloud, i);
            model_LAB.push_back(lab_model);
        }


        /*pcl::PointCloud<pcl::PointXYZL>::Ptr same_colors;
        std::vector<std::vector<int> > grouped_by_color;
        std::vector<int> point_index_to_bucket;
        groupSimilarColors<PointT>(model_cloud, model_LAB, same_colors, grouped_by_color, point_index_to_bucket);

        grouped_by_color_.insert(std::make_pair(all_models->at(m)->id_, grouped_by_color));
        same_colors_.insert(std::make_pair(all_models->at(m)->id_, same_colors));*/
    }

#ifdef VISUALIZE_
    pcl::visualization::PCLVisualizer vis ("Recognition results");
    int v1, v2, v3, v4;
    vis.createViewPort (0.0, 0.0, 0.25, 1.0, v1);
    vis.createViewPort (0.25, 0, 0.5, 1.0, v2);
    vis.createViewPort (0.5, 0, 0.75, 1.0, v3);
    vis.createViewPort (0.75, 0, 1.0, 1.0, v4);
#endif

    for(size_t i=0; i < files_to_recognize.size(); i++)
    {
        std::cout << files_to_recognize[i] << std::endl;
        typename pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (files_to_recognize[i], *scene);

        /*if(scene->isOrganized()) //ATTENTION!
        {
            pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
            ne.setRadiusSearch(0.02f);
            ne.setInputCloud (scene);
            ne.compute (*normal_cloud);

            faat_pcl::utils::noise_models::NguyenNoiseModel<PointT> nm;
            nm.setInputCloud(scene);
            nm.setInputNormals(normal_cloud);
            nm.setLateralSigma(0.001f);
            nm.setMaxAngle(70.f);
            nm.setUseDepthEdges(true);
            nm.compute();
            std::vector<float> weights;
            nm.getWeights(weights);
            nm.getFilteredCloudRemovingPoints(scene, 0.9f);
        }*/

        {
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler (scene);
            vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_z_coloured", v1);
        }

        scene_LAB_values_.resize(scene->points.size());

        for(size_t ii=0; ii < scene->points.size(); ii++)
        {
            bool exists_s;
            float rgb_s;

            typedef pcl::PointCloud<PointT> CloudS;
            typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;

            pcl::for_each_type<FieldListS> (
                        pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene->points[ii],"rgb", exists_s, rgb_s)
                        );

            if (exists_s)
            {

                uint32_t rgb = *reinterpret_cast<int*> (&rgb_s);
                unsigned char rs = (rgb >> 16) & 0x0000ff;
                unsigned char gs = (rgb >> 8) & 0x0000ff;
                unsigned char bs = (rgb) & 0x0000ff;

                float LRefs, aRefs, bRefs;

                RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
                LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                scene_LAB_values_[ii] = (Eigen::Vector3f(LRefs, aRefs, bRefs));
            }
        }

        std::string file_to_recognize(files_to_recognize[i]);
        boost::replace_all (file_to_recognize, scene_file, "");
        boost::replace_all (file_to_recognize, ".pcd", "");
        std::string id_1 = file_to_recognize;

        or_eval.visualizeGroundTruth(vis, id_1, v2, false);

        boost::shared_ptr<std::vector<ModelTPtr> > results_eval;
        results_eval.reset(new std::vector<ModelTPtr>);

        boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms_eval;
        transforms_eval.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

        or_eval.getGroundTruthModelsAndPoses(id_1, results_eval, transforms_eval);
        float resolution = -1;
        float threshold = 0.01f;

        for(size_t k=0; k < results_eval->size(); k++)
        {
            std::stringstream model_name;
            model_name << "model" << k;
            ModelTPtr model = results_eval->at(k);
            ConstPointInTPtr model_cloud = model->getAssembled (resolution);

            PointInTPtr model_cloud_trans(new pcl::PointCloud<PointT>);
            pcl::transformPointCloud(*model_cloud, *model_cloud_trans, transforms_eval->at(k));

            pcl::PointCloud<pcl::PointXYZL>::Ptr faces = model->getAssembledSmoothFaces(resolution);
            pcl::PointCloud<pcl::PointXYZL>::Ptr faces_aligned(new pcl::PointCloud<pcl::PointXYZL>);
            pcl::transformPointCloud (*faces, *faces_aligned, transforms_eval->at (k));

            pcl::PointCloud<pcl::PointXYZI>::Ptr variance_L(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZI>::Ptr variance_AB(new pcl::PointCloud<pcl::PointXYZI>);
            learnColorVariance<PointT>(model_cloud_trans, scene, faces_aligned, variance_L, variance_AB, model->id_, threshold, true);
            //learnColorVariance<PointT>(model_cloud_trans, model_cloud_trans, model->id_, 0.005f, false);

            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> handler(variance_L, "intensity");
            vis.addPointCloud<pcl::PointXYZI>(variance_L, handler, model_name.str(), v3);

            {
                model_name << "AB_";
                pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> handler(variance_AB, "intensity");
                vis.addPointCloud<pcl::PointXYZI>(variance_L, handler, model_name.str(), v4);
            }
        }

        vis.spin ();
        vis.removeAllPointClouds();
    }
}

typedef pcl::ReferenceFrame RFType;

int CG_SIZE_ = 3;
float CG_THRESHOLD_ = 0.005f;


int
main (int argc, char ** argv)
{
    std::string path = "";
    std::string pcd_file = "";

    pcl::console::parse_argument (argc, argv, "-specify_color", specify_color_);
    pcl::console::parse_argument (argc, argv, "-scene_step", SCENE_STEP_);
    pcl::console::parse_argument (argc, argv, "-models_dir", path);
    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-model_scale", model_scale);
    pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);

    MODELS_DIR_FOR_VIS_ = path;

    pcl::console::parse_argument (argc, argv, "-models_dir_vis", MODELS_DIR_FOR_VIS_);
    pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);

    MODELS_DIR_ = path;

    if (pcd_file.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing mians scenes using the -mians_scenes_dir [dir] option\n");
        return -1;
    }

    if (path.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing the models of mian dataset using the -models_dir [dir] option\n");
        return -1;
    }

    boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
            > source (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
    source->setPath (MODELS_DIR_);
    source->setLoadViews (false);
    source->setModelScale(model_scale);
    source->setLoadIntoMemory(false);
    std::string test = "irrelevant";
    std::cout << "calling generate" << std::endl;
    source->setExtension("pcd");
    source->generate (test);
    recognizeAndVisualize<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> (source, pcd_file, "pcd");
}
