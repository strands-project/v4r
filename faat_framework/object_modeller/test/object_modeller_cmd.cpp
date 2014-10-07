
// object modeller
#include "module.h"
#include "result.h"
#include "inputModule.h"
#include "outputModule.h"
#include "pipeline.h"
#include "config.h"

#include "output/windowedPclRenderer.h"

#include "pipelineFactory.h"

using namespace object_modeller;

// helper methods
Config::Ptr parseCommandLineArgs(int argc, char **argv);

int runPipeline(Config::Ptr config);
//int runMultiSequenceAlignment(Config::Ptr config);

/******************************************************************
 * MAIN
 */
int main(int argc, char *argv[] )
{
    Config::Ptr config = parseCommandLineArgs(argc, argv);

    //return runMultiSequenceAlignment(config);
    return runPipeline(config);
}

/*
int runMultiSequenceAlignment(Config &config)
{
    std::cout << "run sequence alignment" << std::endl;

    reader::FileReader reader;
    reader::FileReader model_reader("modelreader");
    reader::PoseReader pose_reader;
    util::Transform t;
    util::DistanceFilter filter;
    util::MultiplyMatrixSingle multiplySingle;
    util::ConvertPointCloud convert_pointcloud;
    util::Mask<pcl::PointXYZRGB> mask;
    segmentation::DominantPlaneExtraction dominant_plane_extraction;
    modelling::NmBasedCloudIntegrationMultiSeq nm_based_cloud_integration_ms;
    multisequence::SiftFeatureMatcher siftFeatureMatcher;
    boost::shared_ptr<output::Renderer> base_renderer;
    base_renderer.reset( new output::WindowedPclRenderer());
    output::PointCloudRenderer<pcl::PointXYZRGB> renderer_xyz(base_renderer);
    output::PointCloudRenderer<pcl::PointXYZRGBNormal> renderer_xyzn(base_renderer);

    util::NormalEstimationOmp normal_estimation;
    util::NguyenNoiseWeights weights_calculation;

    Pipeline pipeline(base_renderer.get(), config);

    reader::FileReader::ResultType *pointclouds_input = pipeline.addIn(&reader);
    reader::FileReader::ResultType *model = pipeline.addIn(&model_reader);
    reader::PoseReader::ResultType *poses = pipeline.addIn(&pose_reader);

    util::DistanceFilter::ResultType *filtered = pipeline.addInOut(&filter, pointclouds_input);
    pipeline.addOut(&renderer_xyz, filtered, new CustomResult<std::string>("initial "), new CustomResult<bool>(false));

    util::Transform::ResultType *transformed = pipeline.addInOut(&t, filtered, poses);

    pipeline.addOut(&renderer_xyz, transformed, new CustomResult<std::string>("transformed"), new CustomResult<bool>(false));


    segmentation::DominantPlaneExtraction::ResultType *indices = pipeline.addInOut(&dominant_plane_extraction, filtered);

    // filter indices and render result
    util::Mask<pcl::PointXYZRGB>::ResultType *pointclouds_segmented = pipeline.addInOut(&mask, transformed, indices);
    pipeline.addOut(&renderer_xyz, pointclouds_segmented, new CustomResult<std::string>("Segmentation output"), new CustomResult<bool>(false));



    pipeline.addOut(&renderer_xyz, model, new CustomResult<std::string>("Model"), new CustomResult<bool>(false));

    util::ConvertPointCloud::ResultType *model2 = pipeline.addInOut(&convert_pointcloud, model);

    pipeline.addOut(&renderer_xyzn, model2, new CustomResult<std::string>("SIFT feature matcher output"), new CustomResult<bool>(false));

    multisequence::SiftFeatureMatcher::ResultType *multiseqPose = pipeline.addInOut(&siftFeatureMatcher, filtered, poses, indices, model2);
    poses = pipeline.addInOut(&multiplySingle, poses, multiseqPose);

    util::NormalEstimationOmp::ResultType *normals = pipeline.addInOut(&normal_estimation, filtered);
    util::NguyenNoiseWeights::ResultType *weights = pipeline.addInOut(&weights_calculation, filtered, normals);

    modelling::NmBasedCloudIntegration::ResultType *final = pipeline.addInOut(&nm_based_cloud_integration_ms, filtered, poses, indices, normals, weights);
    pipeline.addOut(&renderer_xyzn, final, new CustomResult<std::string>("SIFT feature matcher output"), new CustomResult<bool>(true));

    pipeline.process();
}
*/

static void loop(Pipeline::Ptr pipeline);

void loop(Pipeline::Ptr pipeline)
{
    pipeline->process(true);
}

int runPipeline(Config::Ptr config)
{
    output::Renderer::Ptr renderer(new output::WindowedPclRenderer());

    PipelineFactory factory;
    Pipeline::Ptr pipeline = factory.create(factory.getPipelines().front(), config, renderer);


    boost::thread *process_thread = new boost::thread(&loop, pipeline);
    renderer->loop();
    process_thread->join();

    return 0;
}

/**
 * setup command line args
 */
Config::Ptr parseCommandLineArgs(int argc, char **argv)
{
    if (argc > 1)
    {
        // ignore first arg (filename)

        // second arg is config file name
        std::string configPath(argv[1]);

        std::cout << "filename: " << configPath << std::endl;

        Config::Ptr config(new Config());
        config->loadFromFile(configPath);

        for (int i=2;i<argc;i++)
        {
            std::string arg(argv[i]);
            boost::algorithm::trim_left_if(arg, boost::algorithm::is_any_of("-"));

            std::vector<std::string> result;
            boost::algorithm::split(result, arg, boost::algorithm::is_any_of("="));

            // std::cout << result[0] << " = " << result[1] << std::endl;

            config->overrideParameter(result[0], result[1]);
        }

        config->printConfig();

        return config;
    }
    else
    {
        std::cout << "Usage: " << std::endl;
    }
}
