#define BOOST_NO_SCOPED_ENUMS
#define BOOST_NO_CXX11_SCOPED_ENUMS

#include <pcl/point_cloud.h>

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/registered_views_source.h>
#include <v4r/recognition/local_recognizer.h>

#include <boost/any.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <fstream>

#ifndef HAVE_SIFTGPU
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace po = boost::program_options;
namespace bf = boost::filesystem;


class EvalPartialModelRecognizer
{

private:
    typedef pcl::PointXYZRGB PointT;
    std::string in_model_dir_;
    std::string in_training_dir_;
    std::string out_tmp_model_dir_;
    std::string out_tmp_training_dir_;
    std::string out_results_;
    std::vector<std::string> model_list;


public:
    class Parameter
    {
    public:
        int normal_method_;
        float vox_res_;

        Parameter
        ( int normal_method = 2,
          float vox_res = 0.01f)
            : normal_method_ (normal_method),
              vox_res_ (vox_res)
        {}
    }param_;

    EvalPartialModelRecognizer(const Parameter &p = Parameter())
    {
        param_ = p;
        out_tmp_model_dir_ = "/tmp/models/";
        out_tmp_training_dir_ = "/tmp/training_dir/";
        out_results_ = "/tmp/rec_results";
    }


    void
    initialize(int argc, char ** argv)
    {
        po::options_description desc("Evaluation of recognition of partial models\n**Allowed options");
        desc.add_options()
                ("help,h", "produce help message")
                ("in_model_dir,m", po::value<std::string>(&in_model_dir_)->required(), "input model directory")
                ("in_training_dir,t", po::value<std::string>(&in_training_dir_)->required(), "input training directory")
        ;
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return;
        }

        try
        {
            po::notify(vm);
        }
        catch(std::exception& e)
        {
            std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
            return;
        }
    }

    void
    eval()
    {
        v4r::NMBasedCloudIntegration<pcl::PointXYZRGB>::Parameter nm_int_param;
        nm_int_param.final_resolution_ = 0.002f;
        nm_int_param.min_points_per_voxel_ = 1;
        nm_int_param.min_weight_ = 0.5f;
        nm_int_param.octree_resolution_ = 0.002f;
        nm_int_param.threshold_ss_ = 0.01f;

        v4r::noise_models::NguyenNoiseModel<pcl::PointXYZRGB>::Parameter nm_param;

        v4r::io::createDirIfNotExist(out_results_);

        v4r::io::getFilesInDirectory(in_model_dir_, model_list, "", ".*.pcd", false);
        std::sort(model_list.begin(), model_list.end());
        for (size_t replaced_m_id=0; replaced_m_id<model_list.size(); replaced_m_id++)
        {
            const std::string replaced_model = model_list [replaced_m_id];

            bf::remove_all(bf::path(out_tmp_training_dir_));
            bf::remove_all(bf::path(out_tmp_model_dir_));
            v4r::io::createDirIfNotExist(out_tmp_training_dir_);
            v4r::io::createDirIfNotExist(out_tmp_model_dir_);

            // copy all models from model database which are not the one to be tested
            for (size_t m_id=0; m_id<model_list.size(); m_id++)
            {
                if( m_id == replaced_m_id )
                    continue;
                else
                {
                    const std::string model_name = model_list[m_id];
                    bf::copy_file( in_model_dir_ + "/" + model_name, out_tmp_model_dir_ + "/" + model_name );
                    bf::create_directory_symlink(in_training_dir_ + "/" + model_name, out_tmp_training_dir_ + "/" + model_name);
//                    v4r::io::copyDir(in_training_dir_ + "/" + model_name, out_tmp_training_dir_ + "/" + model_name);
                }
            }


            // LOAD ALL DATA FOR THE MODEL TO BE REPLACED BY A PARTIAL MODEL

            std::vector<std::string> training_views;
            v4r::io::getFilesInDirectory(in_training_dir_ + "/" + replaced_model, training_views, "", ".*cloud.*.pcd");
            std::sort(training_views.begin(), training_views.end());

            std::vector<pcl::PointCloud<PointT>::Ptr > training_clouds ( training_views.size() );
            std::vector<pcl::PointCloud<pcl::Normal>::Ptr > normal_clouds ( training_views.size() );
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras ( training_views.size() );
            std::vector<std::vector<int> > obj_indices ( training_views.size() );
            std::vector<pcl::PointCloud<IndexPoint>::Ptr > obj_indices_cloud ( training_views.size() );
            std::vector<std::vector<float> > weights ( training_views.size() );
            std::vector<std::vector<float> > sigmas ( training_views.size() );

            const size_t num_training_views = training_views.size();

            for(size_t v_id = 0; v_id < num_training_views; v_id++)
            {
                const std::string training_view = in_training_dir_ + "/" + replaced_model + "/" + training_views[v_id];

                training_clouds[v_id].reset( new pcl::PointCloud<PointT>);
                normal_clouds[v_id].reset( new pcl::PointCloud<pcl::Normal>);
                pcl::io::loadPCDFile ( training_view, *training_clouds[v_id] );
                std::string path_pose ( training_view );
                boost::replace_last (path_pose, "cloud", "pose");
                boost::replace_last (path_pose, ".pcd", ".txt");
                v4r::io::readMatrixFromFile ( path_pose, cameras[v_id]);

                std::string path_obj_indices (training_view);
                boost::replace_last (path_obj_indices, "cloud", "object_indices");

                obj_indices_cloud[v_id].reset (new pcl::PointCloud<IndexPoint>);
                pcl::io::loadPCDFile (path_obj_indices, *obj_indices_cloud[v_id]);
                obj_indices[v_id].resize(obj_indices_cloud[v_id]->points.size());
                for(size_t kk=0; kk < obj_indices_cloud[v_id]->points.size(); kk++)
                    obj_indices[v_id][kk] = obj_indices_cloud[v_id]->points[kk].idx;

                v4r::computeNormals<PointT>( training_clouds[v_id], normal_clouds[v_id], param_.normal_method_);

                v4r::noise_models::NguyenNoiseModel<PointT> nm;
                nm.setInputCloud ( training_clouds[v_id] );
                nm.setInputNormals ( normal_clouds[v_id] );
                nm.setLateralSigma(0.001);
                nm.setMaxAngle(60.f);
                nm.setUseDepthEdges(true);
                nm.compute();
                nm.getWeights( weights[ v_id ] );
                sigmas[ v_id ] = nm.getSigmas();
            }

            // just for computing the total number of points visible if all training view were considered
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            v4r::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration (nm_int_param);
            nmIntegration.setInputClouds( training_clouds );
            nmIntegration.setWeights( weights );
            nmIntegration.setSigmas( sigmas );
            nmIntegration.setTransformations( cameras );
            nmIntegration.setInputNormals( normal_clouds );
            nmIntegration.setIndices( obj_indices );
            nmIntegration.compute( octree_cloud );
            pcl::PointCloud<PointT> cloud_filtered;
            pcl::VoxelGrid<PointT > sor;
            sor.setInputCloud (octree_cloud);
            sor.setLeafSize ( param_.vox_res_, param_.vox_res_, param_.vox_res_);
            sor.filter ( cloud_filtered );
            size_t total_points = cloud_filtered.points.size();
            size_t total_points_oc = octree_cloud->points.size();

            const std::string out_tmp_training_dir_replaced_model = out_tmp_training_dir_ + "/" + replaced_model;

            size_t eval_id = 0;

            // now create partial model from successive training views
            for (size_t num_used_v = 1; num_used_v < num_training_views; num_used_v++)
            {
                std::vector<pcl::PointCloud<PointT>::Ptr > training_clouds_used ( num_used_v );
                std::vector<pcl::PointCloud<pcl::Normal>::Ptr > normal_clouds_used ( num_used_v );
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras_used ( num_used_v );
                std::vector<std::vector<int> > obj_indices_used ( num_used_v );
                std::vector<std::vector<float> > weights_used ( num_used_v );
                std::vector<std::vector<float> > sigmas_used ( num_used_v );

                for (size_t start_v = 0; start_v < num_training_views; start_v++)
                {
                    boost::filesystem::remove_all( out_tmp_training_dir_replaced_model );
                    v4r::io::createDirIfNotExist( out_tmp_training_dir_replaced_model );

                    for (size_t v_id_rel=0; v_id_rel<num_used_v; v_id_rel++)
                    {
                        size_t v_id = ( start_v + v_id_rel ) % num_training_views;

                        training_clouds_used [ v_id_rel ] = training_clouds [ v_id ];
                        normal_clouds_used [ v_id_rel ] = normal_clouds [ v_id ];
                        cameras_used [ v_id_rel ] = cameras [ v_id ];
                        obj_indices_used [ v_id_rel ] = obj_indices [ v_id ];
                        weights_used [ v_id_rel ] = weights [ v_id ];
                        sigmas_used [ v_id_rel ] = sigmas [ v_id ];

                        const std::string out_cloud_file = out_tmp_training_dir_replaced_model + "/" + training_views[v_id];
                        std::string path_pose ( out_cloud_file );
                        boost::replace_last (path_pose, "cloud", "pose");
                        boost::replace_last (path_pose, ".pcd", ".txt");
                        std::string path_obj_indices ( out_cloud_file );
                        boost::replace_last (path_obj_indices, "cloud", "object_indices");

                        pcl::io::savePCDFileBinary ( out_cloud_file, *training_clouds [ v_id ] );
                        v4r::io::writeMatrixToFile ( path_pose, cameras [ v_id ] );
                        pcl::io::savePCDFileBinary ( path_obj_indices, *obj_indices_cloud[ v_id ] );
                    }

                    nmIntegration.setInputClouds( training_clouds_used );
                    nmIntegration.setWeights( weights_used );
                    nmIntegration.setSigmas( sigmas_used );
                    nmIntegration.setTransformations( cameras_used );
                    nmIntegration.setInputNormals( normal_clouds_used );
                    nmIntegration.setIndices( obj_indices_used );
                    nmIntegration.compute( octree_cloud );

                    sor.setInputCloud (octree_cloud);
                    sor.setLeafSize ( param_.vox_res_, param_.vox_res_, param_.vox_res_);
                    sor.filter ( cloud_filtered );

                    pcl::io::savePCDFileBinary(out_tmp_model_dir_ + "/" + replaced_model, *octree_cloud);

                    std::cout << cloud_filtered.points.size() << " / " << total_points << " visible."
                              << static_cast<float>(cloud_filtered.points.size()) / total_points << " "
                              << static_cast<float>(octree_cloud->points.size()) / total_points_oc <<std::endl;

                    rec();
                    std::stringstream result_fn; result_fn << out_results_ << "/result_" << eval_id << ".txt";
                    ofstream f(result_fn.str().c_str());
                    f << replaced_model << " " << num_used_v << " " << num_training_views << " " <<
                         cloud_filtered.points.size() << " " << total_points <<std::endl;
                    f.close();
                    eval_id++;
                }
            }
            exit(0);
        }
    }

    void rec()
    {

    }
};


int
main (int argc, char ** argv)
{
    srand (time(NULL));
    EvalPartialModelRecognizer r;
    r.initialize(argc,argv);
    r.eval();
    return 0;
}
