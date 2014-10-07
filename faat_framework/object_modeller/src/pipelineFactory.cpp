
#include "pipelineFactory.h"

#include "pipeline.h"

#include "reader/fileReader.h"
#include "reader/cameraReader.h"
#include "reader/turntableReader.h"
#include "reader/trackingCameraReader.h"
#include "reader/poseReader.h"

#include "segmentation/dominantPlaneExtraction.h"

#include "util/distanceFilter.h"
#include "util/boxFilter.h"

#include "util/transform.h"
#include "util/mask.h"
#include "util/vectorMask.h"
#include "util/multiplyMatrix.h"

#include "util/normalEstimationOmp.h"
#include "util/integralImageNormalEstimation.h"
#include "util/nguyenNoiseWeights.h"

#include "registration/globalRegistration.h"
#include "registration/cameraTracker.h"
#include "registration/checkerboard.h"

#include "modelling/nmBasedCloudIntegration.h"
#include "modelling/poissonReconstruction.h"

#include "multisequence/siftFeatureMatcher.h"

#include "output/pointCloudWriter.h"
#include "output/indicesWriter.h"
#include "output/posesWriter.h"

#include "texturing/pclTexture.h"
#include "texturing/shadingTexture.h"

namespace object_modeller
{


    Pipeline::Ptr PipelineFactory::create(std::string name, Config::Ptr config, output::Renderer::Ptr renderer)
    {
        Pipeline::Ptr pipeline(new object_modeller::Pipeline(renderer, config));
        MethodPointer init = functions[name];
        (this->*init)(pipeline, renderer);
        return pipeline;
    }

    void PipelineFactory::initTexturing(Pipeline::Ptr pipeline, output::Renderer::Ptr renderer)
    {
        object_modeller::reader::FileReader<pcl::PointXYZRGB> *input_reader = new object_modeller::reader::FileReader<pcl::PointXYZRGB>();
        object_modeller::reader::FileReader<pcl::PointXYZRGBNormal> *model_reader = new object_modeller::reader::FileReader<pcl::PointXYZRGBNormal>("modelReader");
        object_modeller::reader::PoseReader *pose_reader = new object_modeller::reader::PoseReader();

        object_modeller::reader::FileReader<pcl::PointXYZRGB>::ResultType pointclouds_input = pipeline->addIn(input_reader);
        object_modeller::reader::FileReader<pcl::PointXYZRGBNormal>::ResultType model = pipeline->addIn(model_reader);
        object_modeller::reader::PoseReader::ResultType poses = pipeline->addIn(pose_reader);

        // poisson
        object_modeller::modelling::PoissonReconstruction *poisson_reconstruction = new object_modeller::modelling::PoissonReconstruction();
        object_modeller::modelling::PoissonReconstruction::ResultType mesh = pipeline->addInOut(poisson_reconstruction, model);

        // texturing
        object_modeller::texturing::PclTexture *pcl_texture = new object_modeller::texturing::PclTexture();
        object_modeller::texturing::ShadingTexture *shading_texture = new object_modeller::texturing::ShadingTexture();
        object_modeller::texturing::PclTexture::ResultType textured_mesh = pipeline->beginChoice<object_modeller::texturing::PclTexture::ReturnType>("texturing", "Texturing");
        pipeline->addInOut(shading_texture, mesh, model, poses);
        pipeline->addInOut(pcl_texture, mesh, pointclouds_input, poses);
        pipeline->endChoice();

        // output
        output::PointCloudWriter<pcl::PointXYZRGBNormal> *model_writer = new output::PointCloudWriter<pcl::PointXYZRGBNormal>("modelWriter");
        pipeline->addOut(model_writer, model);
    }

    void PipelineFactory::initStandard(Pipeline::Ptr pipeline, output::Renderer::Ptr renderer)
    {
        object_modeller::util::Transform *transform = new object_modeller::util::Transform();
        object_modeller::util::MultiplyMatrix *multiply = new object_modeller::util::MultiplyMatrix();
        object_modeller::util::Mask<pcl::PointXYZRGB> *mask = new object_modeller::util::Mask<pcl::PointXYZRGB>();
        object_modeller::util::Mask<pcl::Normal> *mask_normals = new object_modeller::util::Mask<pcl::Normal>();
        object_modeller::util::VectorMask<float> *mask_weights = new object_modeller::util::VectorMask<float>();
        object_modeller::registration::GlobalRegistration *global_registration = new object_modeller::registration::GlobalRegistration();

        // input
        object_modeller::registration::CameraTracker *cam_tracker = new object_modeller::registration::CameraTracker();
        object_modeller::reader::FileReader<pcl::PointXYZRGB> *reader = new object_modeller::reader::FileReader<pcl::PointXYZRGB>();
        object_modeller::reader::CameraReader *cam_reader = new object_modeller::reader::CameraReader(renderer);
        object_modeller::reader::TurntableReader *tt_reader = new object_modeller::reader::TurntableReader(renderer);
        object_modeller::reader::TrackingCameraReader *tracking_cam_reader = new object_modeller::reader::TrackingCameraReader(cam_tracker, renderer);

        object_modeller::reader::FileReader<pcl::PointXYZRGB>::ResultType pointclouds_input = pipeline->beginChoice<object_modeller::reader::FileReader<pcl::PointXYZRGB>::ReturnType>("input", "Input");
        pipeline->addIn(reader);
        pipeline->addIn(cam_reader);
        pipeline->addIn(tt_reader);
        pipeline->addIn(tracking_cam_reader);
        pipeline->endChoice();

        // filtering
        object_modeller::util::DistanceFilter *distance_filter = new object_modeller::util::DistanceFilter();
        object_modeller::util::BoxFilter *box_filter = new object_modeller::util::BoxFilter();

        object_modeller::util::DistanceFilter::ResultType pointclouds_filtered = pipeline->beginChoice<object_modeller::util::DistanceFilter::ReturnType>("filtering", "Filtering");
        pipeline->addInOut(distance_filter, pointclouds_input);
        pipeline->addInOut(box_filter, pointclouds_input);
        pipeline->endChoice();

        // registration
        object_modeller::registration::CheckerboardRegistration *checkerboard_reg = new object_modeller::registration::CheckerboardRegistration();

        object_modeller::registration::CameraTracker::ResultType poses = pipeline->beginChoice<object_modeller::registration::CameraTracker::ReturnType>("registration", "Registration");
        pipeline->addInOut(cam_tracker, pointclouds_filtered);
        pipeline->addInOut(checkerboard_reg, pointclouds_filtered);
        pipeline->endChoice();

        // apply transformation from registration
        object_modeller::util::Transform::ResultType pointclouds_transformed = pipeline->addInOut(transform, pointclouds_filtered, poses);

        //segmentation
        object_modeller::segmentation::DominantPlaneExtraction *dominant_plane_extraction = new object_modeller::segmentation::DominantPlaneExtraction();

        object_modeller::segmentation::DominantPlaneExtraction::ResultType indices = pipeline->addInOut(dominant_plane_extraction, pointclouds_filtered);

        // apply indices from segmentation to input
        object_modeller::util::Mask<pcl::PointXYZRGB>::ResultType pointclouds_segmented = pipeline->addInOut(mask, pointclouds_transformed, indices);

        // estimate normals and weights
        object_modeller::util::NormalEstimationOmp *normal_estimation = new object_modeller::util::NormalEstimationOmp();
        //object_modeller::util::IntegralImageNormalEstimation *normal_estimation = new object_modeller::util::IntegralImageNormalEstimation();
        object_modeller::util::NguyenNoiseWeights *weights_calculation = new object_modeller::util::NguyenNoiseWeights();

        object_modeller::util::NormalEstimationOmp::ResultType normals = pipeline->addInOut(normal_estimation, pointclouds_filtered);
        object_modeller::util::NguyenNoiseWeights::ResultType weights = pipeline->addInOut(weights_calculation, pointclouds_filtered, normals);

        // global registration
        pipeline->beginOptional("multiview", "Multiview");
        // TODO: use already calculated normals and weights

        util::NormalEstimationOmp::ResultType normals_transformed = pipeline->addInOut(normal_estimation, pointclouds_transformed);
        util::NguyenNoiseWeights::ResultType weights_transformed = pipeline->addInOut(weights_calculation, pointclouds_transformed, normals);

        normals_transformed = pipeline->addInOut(mask_normals, normals_transformed, indices);
        weights_transformed = pipeline->addInOut(mask_weights, weights_transformed, indices);

        object_modeller::registration::GlobalRegistration::ResultType global_reg_poses = pipeline->addInOut(global_registration, pointclouds_segmented, normals_transformed, weights_transformed);
        poses = pipeline->addInOut(multiply, global_reg_poses, poses);
        pointclouds_segmented = pipeline->addInOut(mask, pointclouds_filtered, indices);
        pointclouds_transformed = pipeline->addInOut(transform, pointclouds_segmented, poses);
        pipeline->endOptional();

        // nm based
        object_modeller::modelling::NmBasedCloudIntegration *nm_based_cloud_integration = new object_modeller::modelling::NmBasedCloudIntegration();
        object_modeller::modelling::NmBasedCloudIntegration::ResultType model = pipeline->addInOut(nm_based_cloud_integration, pointclouds_filtered, poses, indices, normals, weights);

        // multi sequence alignment
        object_modeller::modelling::NmBasedCloudIntegrationMultiSeq *nm_based_cloud_integration_ms = new object_modeller::modelling::NmBasedCloudIntegrationMultiSeq();
        multisequence::SiftFeatureMatcher *siftFeatureMatcher = new multisequence::SiftFeatureMatcher();
        util::MultiplyMatrixSingle *multiplySingle = new util::MultiplyMatrixSingle();

        multisequence::SiftFeatureMatcher::ResultType multiseqPose = pipeline->addInOut(siftFeatureMatcher, pointclouds_input, poses, indices, model);
        poses = pipeline->addInOut(multiplySingle, poses, multiseqPose);
        model = pipeline->addInOut(nm_based_cloud_integration_ms, pointclouds_filtered, poses, indices, normals, weights);

        // poisson
        object_modeller::modelling::PoissonReconstruction *poisson_reconstruction = new object_modeller::modelling::PoissonReconstruction();
        object_modeller::modelling::PoissonReconstruction::ResultType mesh = pipeline->addInOut(poisson_reconstruction, model);

        // texturing
        object_modeller::texturing::PclTexture *pcl_texture = new object_modeller::texturing::PclTexture();
        object_modeller::texturing::ShadingTexture *shading_texture = new object_modeller::texturing::ShadingTexture();
        object_modeller::texturing::PclTexture::ResultType textured_mesh = pipeline->beginChoice<object_modeller::texturing::PclTexture::ReturnType>("texturing", "Texturing");
        pipeline->addInOut(shading_texture, mesh, model, poses);
        pipeline->addInOut(pcl_texture, mesh, pointclouds_input, poses);
        pipeline->endChoice();

        // output
        output::PointCloudWriter<pcl::PointXYZRGB> *pointcloud_writer = new output::PointCloudWriter<pcl::PointXYZRGB>();
        output::IndicesWriter *indices_writer = new output::IndicesWriter();
        output::PosesWriter *poses_writer = new output::PosesWriter();
        output::PointCloudWriter<pcl::PointXYZRGBNormal> *model_writer = new output::PointCloudWriter<pcl::PointXYZRGBNormal>("modelWriter");

        pipeline->addOut(poses_writer, poses);
        pipeline->addOut(indices_writer, indices);
        pipeline->addOut(pointcloud_writer, pointclouds_input);
        pipeline->addOut(model_writer, model);
    }
}
