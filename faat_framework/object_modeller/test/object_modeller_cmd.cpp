
// object modeller
#include "module.h"
#include "result.h"
#include "inputModule.h"
#include "outputModule.h"
#include "pipeline.h"
#include "config.h"

#include "reader/fileReader.h"

#include "registration/cameraTracker.h"
#include "registration/checkerboard.h"
#include "registration/globalRegistration.h"

#include "segmentation/dominantPlaneExtraction.h"

#include "output/tgRenderer.h"
#include "output/pclRenderer.h"
#include "output/pointCloudWriter.h"
#include "output/indicesWriter.h"
#include "output/posesWriter.h"
#include "output/meshRenderer.h"
#include "output/renderer.h"

#include "util/transform.h"
#include "util/mask.h"
#include "util/multiplyMatrix.h"
#include "util/distanceFilter.h"
#include "util/normalEstimationOmp.h"
#include "util/integralImageNormalEstimation.h"
#include "util/nguyenNoiseWeights.h"
#include "util/vectorMask.h"

#include "modelling/nmBasedCloudIntegration.h"
#include "modelling/poissonReconstruction.h"

using namespace object_modeller;

// helper methods
Config parseCommandLineArgs(int argc, char **argv);


/******************************************************************
 * MAIN
 */
int main(int argc, char *argv[] )
{
    Config config = parseCommandLineArgs(argc, argv);

    bool step = config.getBool("pipeline.step");

    // input reader
    reader::FileReader reader;

    // point cloud utility
    util::DistanceFilter distance_filter;
    util::Transform transform;
    util::Mask<pcl::PointXYZRGB> mask;
    util::Mask<pcl::Normal> mask_normals;
    util::VectorMask<float> mask_weights;
    util::MultiplyMatrix multiply;

    util::NormalEstimationOmp normal_estimation;
    util::IntegralImageNormalEstimation normal_estimation_fast;
    util::NguyenNoiseWeights weights_calculation;

    // registration
    registration::CameraTracker camera_tracker;
    registration::CheckerboardRegistration checkerboard;

    registration::GlobalRegistration global_registration;

    //segmentation
    segmentation::DominantPlaneExtraction dominant_plane_extraction;

    //modelling
    modelling::NmBasedCloudIntegration nm_based_cloud_integration;
    modelling::PoissonReconstruction poisson_reconstruction;

    //renderer
    boost::shared_ptr<output::Renderer> renderer;

    //output
    output::PointCloudWriter pointcloud_writer;
    output::IndicesWriter indices_writer;
    output::PosesWriter poses_writer;

    if (config.getInt("pipeline.renderer") == 0)
    {
        renderer.reset( new output::PclRenderer());
    } else {
        renderer.reset( new output::TomGineRenderer());
    }

    output::MeshRenderer mesh_renderer(renderer);


    // setup pipeline
    Pipeline pipeline(config);

    //read from file
    reader::FileReader::ResultType *pointclouds_input = pipeline.addIn(&reader);

    // filter far points
    util::DistanceFilter::ResultType *pointclouds_filtered = pipeline.addInOut(&distance_filter, pointclouds_input);

    registration::CameraTracker::ResultType *poses;

    // calculate poses
    if (config.getInt("pipeline.registrationType") == 0)
    {
        // checkerboard
        poses = pipeline.addInOut(&checkerboard, pointclouds_filtered);
    }
    else
    {
        // camera tracker
        poses = pipeline.addInOut(&camera_tracker, pointclouds_filtered);
    }

    // apply poses and render result
    util::Transform::ResultType *pointclouds_transformed = pipeline.addInOut(&transform, pointclouds_filtered, poses);
    pipeline.addOut(renderer.get(), pointclouds_transformed, new Result<std::string>("Registration Output"), new Result<bool>(step));

    // segmentation
    segmentation::DominantPlaneExtraction::ResultType *indices = pipeline.addInOut(&dominant_plane_extraction, pointclouds_filtered);

    // filter indices and render result
    util::Mask<pcl::PointXYZRGB>::ResultType *pointclouds_segmented = pipeline.addInOut(&mask, pointclouds_transformed, indices);
    pipeline.addOut(renderer.get(), pointclouds_segmented, new Result<std::string>("Segmentation output"), new Result<bool>(step));

    //global registration
    if (config.getBool("pipeline.enableMultiview"))
    {
        // estimate normals and weights
        util::NormalEstimationOmp::ResultType *normals = pipeline.addInOut(&normal_estimation_fast, pointclouds_transformed);
        util::NguyenNoiseWeights::ResultType *weights = pipeline.addInOut(&weights_calculation, pointclouds_transformed, normals);

        normals = pipeline.addInOut(&mask_normals, normals, indices);
        weights = pipeline.addInOut(&mask_weights, weights, indices);

        registration::GlobalRegistration::ResultType *global_reg_poses = pipeline.addInOut(&global_registration, pointclouds_segmented, normals, weights);

        poses = pipeline.addInOut(&multiply, global_reg_poses, poses);
        pointclouds_segmented = pipeline.addInOut(&mask, pointclouds_filtered, indices);
        pointclouds_transformed = pipeline.addInOut(&transform, pointclouds_segmented, poses);
        pipeline.addOut(renderer.get(), pointclouds_transformed, new Result<std::string>("Global registration output"), new Result<bool>(step));
    }

    // estimate normals and weights
    util::NormalEstimationOmp::ResultType *normals = pipeline.addInOut(&normal_estimation, pointclouds_filtered);
    util::NguyenNoiseWeights::ResultType *weights = pipeline.addInOut(&weights_calculation, pointclouds_filtered, normals);

    // nm based cloud integration
    modelling::NmBasedCloudIntegration::ResultType *model = pipeline.addInOut(&nm_based_cloud_integration, pointclouds_filtered, poses, indices, normals, weights);
    pipeline.addOut(renderer.get(), model, new Result<std::string>("NM based cloud integration output"), new Result<bool>(step));

    // poisson reconstruction
    modelling::PoissonReconstruction::ResultType *mesh = pipeline.addInOut(&poisson_reconstruction, model);
    pipeline.addOut(&mesh_renderer, mesh, new Result<std::string>("Poisson reconstruction output"), new Result<bool>(step));

    // output
    pipeline.addOut(&poses_writer, poses);
    pipeline.addOut(&indices_writer, indices);
    pipeline.addOut(&pointcloud_writer, model);

    pipeline.process();

    return 0;
}

/**
 * setup command line args
 */
Config parseCommandLineArgs(int argc, char **argv)
{
    if (argc > 1)
    {
        // ignore first arg (filename)

        // second arg is config file name
        std::string configPath(argv[1]);

        std::cout << "filename: " << configPath << std::endl;

        Config config(configPath);

        for (int i=2;i<argc;i++)
        {
            std::string arg(argv[i]);
            boost::algorithm::trim_left_if(arg, boost::algorithm::is_any_of("-"));

            std::vector<std::string> result;
            boost::algorithm::split(result, arg, boost::algorithm::is_any_of("="));

            // std::cout << result[0] << " = " << result[1] << std::endl;

            config.overrideParameter(result[0], result[1]);
        }

        config.printConfig();

        return config;
    }
    else
    {
        std::cout << "Usage: " << std::endl;
    }
}
