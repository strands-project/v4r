/******************************************************************************
 * Copyright (c) 2017 Daniel Wolf
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

#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <ctime>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/impl/instantiate.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <v4r/semantic_segmentation/supervoxel_segmentation.h>
#include <v4r/semantic_segmentation/entangled_forest.h>
#include <v4r/semantic_segmentation/entangled_feature_extraction.h>



#define SIGMA_ROTATION              3.141592654 * 10 / 180
#define SIGMA_CAMERA_ROTATION       3.141592654 * 1 / 180
#define SIGMA_CAMERA_HEIGHT         0.05    // 5cm
#define SIGMA_SEEDS                 0.02//0.03
#define SIGMA_SUPERVOXEL_PARAMS     0.01//0.05
#define SIGMA_MERGING_PARAMS        0.01
#define SIGMA_MERGE_PARAM           0.003//0.005

namespace po = boost::program_options;
using namespace boost::posix_time;

using namespace std;
using namespace pcl;
using namespace cv;

// supervoxel settings
float voxelResolution;
float seedResolution;
float colorImportance;
float spatialImportance;
float normalImportance;

float mergingThreshold;
float colorW;
float normalW;
float curvatureW;
float ptplW;

std::string cloudinputdir;
std::string lblcloudinputdir;
std::string indexfile;
std::string outputdir;

unsigned int floorLabel;
unsigned int minFloorPoints;
float floorPlaneInlierMargin;

// if floor plane fit fails
double estimatedCameraHeight = 1.3;  // ~ avg. from NYU v1 good floor fits
string anglefile;

bool cie94;
bool bilateralfilter;
bool outlierfilter;
bool integralImages;

unsigned int augmentations;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr mergedCloud;

vector<double> seedResolutionDev;
vector<double> colorImportanceDev;
vector<double> normalImportanceDev;
vector<double> spatialImportanceDev;
vector<double> mergingThresholdDev;
vector<double> colorWDev;
vector<double> normalWDev;
vector<double> curvatureWDev;
vector<double> ptplWDev;
vector<double> augYawDev;
vector<double> augPitchDev;
vector<double> augRollDev;
vector<double> augCamHeightDev;

static bool parseArgs(int argc, char** argv)
{
    po::options_description general("General options");
    general.add_options()
            ("help,h","")
            ("cloud-input-dir,i", po::value<std::string>(&cloudinputdir), "Pointcloud input directory")
            ("labelcloud-input-dir,l", po::value<std::string>(&lblcloudinputdir), "Labeled pointcloud input directory")
            ("index-file,x", po::value<std::string>(&indexfile), "Pointcloud index file")
            ("output-dir,o", po::value<std::string>(&outputdir), "")
            ("angle-file,g", po::value<std::string>(&anglefile)->default_value("angles"), "Kinect accelerometer angles file")
            ("augmentations,a", po::value<unsigned>(&augmentations)->default_value(0), "Create N augmented versions of the pointcloud")
            ;

    po::options_description filter("Preprocessing options");
    filter.add_options()
            ("bilfilter,b", po::value<bool>(&bilateralfilter)->default_value(true), "Use bilateral filter" )
            ("outlierfilter,u", po::value<bool>(&outlierfilter)->default_value(false), "Use statistical outlier filter" )
            ("integral", po::value<bool>(&integralImages)->default_value(false), "Use integral images for normals")
            ;
    po::options_description sv("Supervoxel parameters");
    sv.add_options()
            ("voxel-resolution,v", po::value<float>(&voxelResolution)->default_value(0.01), "Resolution of voxel grid" )
            ("seed-resolution,r", po::value<float>(&seedResolution)->default_value(0.3), "Resolution of supervoxel seeds" )
            ("color-importance,c", po::value<float>(&colorImportance)->default_value(0.6), "Weight of color features for clustering" )
            ("spatial-importance,s", po::value<float>(&spatialImportance)->default_value(0.3), "Weight of spatial features for clustering" )
            ("normal-importance,n", po::value<float>(&normalImportance)->default_value(1.0), "Weight of normal features for clustering" )
            ("merging-threshold,m", po::value<float>(&mergingThreshold)->default_value(0.06), "Max. dissimilarity measure to merge supervoxels" )
            ("merging-color", po::value<float>(&colorW)->default_value(1.0), "Merge weight for color" )
            ("merging-normals", po::value<float>(&normalW)->default_value(1.0), "Merge weight for normals" )
            ("merging-curvature", po::value<float>(&curvatureW)->default_value(0.0), "Merge weight for curvature" )
            ("merging-ptpl", po::value<float>(&ptplW)->default_value(1.0), "Merge weight for ptpl distance" )
            ("cie94", po::value<bool>(&cie94)->default_value(true), "Use CIE94 color distance" )
            ;

    po::options_description floorfit("Floor fitting parameters");
    floorfit.add_options()
            ("floor-label", po::value<unsigned>(&floorLabel)->default_value(6), "Label ID for floor (1-based)")
            ("plane-inlier-margin,p", po::value<float>(&floorPlaneInlierMargin)->default_value(0.05f), "Inlier margin for plane fit")
            ("min-floor-points,f", po::value<unsigned>(&minFloorPoints)->default_value(5000), "Min. number of available floor points")
            ;

    po::options_description all("");
    all.add(general).add(filter).add(sv).add(floorfit);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(all).run(), vm);

    po::notify(vm);    

    if(vm.count("help") || !vm.count("cloud-input-dir") || !vm.count("labelcloud-input-dir") || !vm.count("index-file") || !vm.count("output-dir") || !vm.count("angle-file"))
    {
        std::cout << "General usage: create_3DEF_trainingdata [options] -i pointcloud_dir -l labeled-pointcloud-dir -x index-file -g camera-angle-file -o output-dir" << std::endl;
        std::cout << all;
        return false;
    }    

    return true;
}

static void LoadAccelerometerPitchAndRoll(string filename, vector<double>& rollAngles, vector<double>& pitchAngles)
{
    rollAngles.clear();
    pitchAngles.clear();

    ifstream ifs(filename.c_str());

    double angle;
    while(ifs >> angle)
    {
        rollAngles.push_back(angle);
        ifs >> angle;
        pitchAngles.push_back(angle);
    }
    ifs.close();
}

static void GetAccelerometerPitchAndRoll(double roll, double pitch, Eigen::Matrix3f& rotationMatrix)
{
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yawAngle(0, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitX());

    Eigen::Quaternion<float> q = rollAngle * yawAngle * pitchAngle;

    rotationMatrix = q.matrix();
}

static bool CalculateHeightPitchAndRoll(pcl::PointCloud<pcl::PointXYZL>::Ptr labelCloud, float& height, Eigen::Matrix3f& rotationMatrix)
{
    pcl::PointIndices::Ptr floorIndices (new pcl::PointIndices);

    for(unsigned int i=0; i<labelCloud->points.size(); ++i)
    {
        if(labelCloud->points[i].label == floorLabel)
            floorIndices->indices.push_back(i);
    }

    if(floorIndices->indices.size() < minFloorPoints)
    {
        std::cout << "Too few floor points in pointcloud! (" << floorIndices->indices.size() << ")" << std::endl;
        return false;
    }

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZL> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (floorPlaneInlierMargin);

    seg.setInputCloud (labelCloud);
    seg.setIndices(floorIndices);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () < minFloorPoints)
    {
        std::cout << "Too few floor plane inliers! ("<< inliers->indices.size() << ")" << std::endl;
        return false;
    }
    if(((double)inliers->indices.size()) / floorIndices->indices.size() < 0.8)
    {
        std::cout << "Too few floor plane inliers compared to outliers! ("<< ((double)inliers->indices.size()) / floorIndices->indices.size() << ")" << std::endl;
        return false;
    }

    height = coefficients->values[3];

    // calculate angles with floor plane
    Eigen::Vector3f vertical;
    vertical << 0, -1, 0;   // normal of groundplane is exactly vertical and upwards
    Eigen::Vector3f plane;
    plane << coefficients->values[0] , coefficients->values[1] , coefficients->values[2];

    // calculate rotation quaternion, then rotation matrix and from there euler angles for roll, pitch and yaw rotation
    Eigen::Quaternion<float> q;
    q.setFromTwoVectors(plane,vertical);
    rotationMatrix = q.toRotationMatrix();

    if(plane.dot(vertical) < 0)
    {
        std::cout << "Floor not correctly detected (dot product = " << plane.dot(vertical) << ")" << std::endl;
        return false;
    }

    return true;
}

static void saveUnaryFeatures(string filename, std::vector< std::vector<double> >& features)
{
    // save converted
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << features;
    ofs.close();
}

static void savePairwiseFeatures(string filename, std::vector<std::vector<std::pair<double, int> > >& features)
{
    // save converted
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << features;
    ofs.close();
}

static void saveLabels(string filename, std::vector< int >& labels)
{
    // save converted
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << labels;
    ofs.close();
}

template <typename PointT>
static void getInlierIndices(const boost::shared_ptr<pcl::PointCloud<PointT> >& cloud, int meanK, double stddevMulThresh, pcl::IndicesPtr inlier_indices)
{
    cout << "Calculating statistical outliers..." << endl;

    // temporary cloud to apply distance compensation
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outlierremoved(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_outlierremoved);

    inlier_indices->clear();

    // ignore all 0.0 values
    pcl::PassThrough<pcl::PointXYZ> limitz2;
    limitz2.setFilterFieldName("z");
    limitz2.setInputCloud(cloud_outlierremoved);
    limitz2.setFilterLimits(0.0f, 0.0f);
    limitz2.setFilterLimitsNegative(true);
    limitz2.filter(*inlier_indices);

    // distance compensation (for limited point density in the distance)
    for(int i=0; i<inlier_indices->size(); ++i)
    {
        int idx = inlier_indices->at(i);
        double z = cloud_outlierremoved->points[idx].z;
        cloud_outlierremoved->points[idx].x /= z;
        cloud_outlierremoved->points[idx].y /= z;
        cloud_outlierremoved->points[idx].z = std::log(z);
    }

    // works only on unorganized data, otherwise statoutlierremover complains
    cloud_outlierremoved->is_dense = false;
    cloud_outlierremoved->height = 1;
    cloud_outlierremoved->width = cloud_outlierremoved->points.size();

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud_outlierremoved);
    sor.setIndices(inlier_indices);
    sor.setMeanK (meanK);
    sor.setStddevMulThresh (stddevMulThresh);
    sor.setKeepOrganized(true);
    sor.filter(*inlier_indices);
}

template <typename PointT>
static void applyBilateralFilter(const boost::shared_ptr<pcl::PointCloud<PointT> >& inputcloud, pcl::IndicesConstPtr indices, double sigmaS, double sigmaR, boost::shared_ptr<pcl::PointCloud<PointT> >& outputcloud)
{
    cout << "Apply bilateral filter..." << endl;
    pcl::FastBilateralFilterOMP<PointT> bf;
    bf.setInputCloud (inputcloud);
    bf.setIndices(indices);
    bf.setSigmaS(sigmaS);
    bf.setSigmaR(sigmaR);
    bf.filter(*outputcloud);
}


template<typename PointT>
static void removeOutliers(const boost::shared_ptr<pcl::PointCloud<PointT> >& inputcloud, pcl::IndicesConstPtr indices, boost::shared_ptr<pcl::PointCloud<PointT> >& outputcloud)
{
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(inputcloud);
    extract.setIndices(indices);
    extract.setNegative(false);
    extract.filter(*outputcloud);
}

static void CreateNRandomDeviations(int N)
{
    seedResolutionDev.assign(N, 0.0);
    colorImportanceDev.assign(N, 0.0);
    normalImportanceDev.assign(N, 0.0);
    spatialImportanceDev.assign(N, 0.0);
    mergingThresholdDev.assign(N, 0.0);
    colorWDev.assign(N, 0.0);
    normalWDev.assign(N, 0.0);
    curvatureWDev.assign(N, 0.0);
    ptplWDev.assign(N, 0.0);
    augYawDev.assign(N, 0.0);
    augPitchDev.assign(N, 0.0);
    augRollDev.assign(N, 0.0);
    augCamHeightDev.assign(N, 0.0);

    boost::mt19937 randomGenerator(std::time(NULL));
    boost::random::normal_distribution<double> randomRot(0.0, SIGMA_ROTATION);
    boost::random::normal_distribution<double> randomCamRot(0.0, SIGMA_CAMERA_ROTATION);
    boost::random::normal_distribution<double> randomCamHeight(0.0, SIGMA_CAMERA_HEIGHT);
    boost::random::normal_distribution<double> randomSeeds(0.0, SIGMA_SEEDS);
    boost::random::normal_distribution<double> randomSupervoxels(0.0, SIGMA_SUPERVOXEL_PARAMS);
    boost::random::normal_distribution<double> randomMergingParams(0.0, SIGMA_MERGING_PARAMS);
    boost::random::normal_distribution<double> randomMerge(0.0, SIGMA_MERGE_PARAM);

    for(int i=0; i<N; ++i)
    {
        seedResolutionDev[i] = randomSeeds(randomGenerator);
        colorImportanceDev[i] = randomSupervoxels(randomGenerator);
        normalImportanceDev[i] = randomSupervoxels(randomGenerator);
        spatialImportanceDev[i] = randomSupervoxels(randomGenerator);
        mergingThresholdDev[i] = randomMerge(randomGenerator);
        colorWDev[i] = randomMergingParams(randomGenerator);
        normalWDev[i] = randomMergingParams(randomGenerator);
        curvatureWDev[i] = randomMergingParams(randomGenerator);
        ptplWDev[i] += randomMergingParams(randomGenerator);
        augYawDev[i] = randomRot(randomGenerator);
        augPitchDev[i] = randomCamRot(randomGenerator);
        augRollDev[i] = randomCamRot(randomGenerator);
        augCamHeightDev[i] = randomCamHeight(randomGenerator);
    }
}


static void createRotatedAugmentation(double roll, double pitch, double yaw,
                               pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotatedInput)
{
    Eigen::Matrix4f rotationMatrix;
    Eigen::Matrix4f ryaw;
    Eigen::Matrix4f rpitch;
    Eigen::Matrix4f rroll;

    rpitch << 1,               0,               0, 0,
              0,  cos(pitch),  sin(pitch), 0,
              0, -sin(pitch),  cos(pitch), 0,
              0, 0, 0, 1;
    rroll << cos(roll), sin(roll),  0, 0,
             -sin(roll), cos(roll), 0, 0,
             0,                         0,  1, 0,
             0, 0, 0, 1;
    ryaw << cos(yaw), 0, -sin(yaw), 0,
            0,            1,             0, 0,
            sin(yaw), 0,  cos(yaw), 0,
            0, 0, 0, 1;

    rotationMatrix = rpitch * rroll * ryaw;

    // rotate input pointcloud
    pcl::transformPointCloud(*input, *rotatedInput, rotationMatrix);
}

static void createOutputDirectories(string clusteringstr, string labelstr, string pairwisestr, string unarystr)
{
    boost::filesystem::path unaryPath(unarystr);
    if(!exists(unaryPath))
        boost::filesystem::create_directories(unaryPath);

    boost::filesystem::path pairwisePath(pairwisestr);
    if(!exists(pairwisePath))
        boost::filesystem::create_directories(pairwisePath);

    boost::filesystem::path labelsPath(labelstr);
    if(!exists(labelsPath))
        boost::filesystem::create_directories(labelsPath);

    boost::filesystem::path clusteringPath(clusteringstr);
    if(!exists(clusteringPath))
        boost::filesystem::create_directories(clusteringPath);
}

static bool loadIndexFile(std::vector<string>& filenames)
{
    string filename;

    // read list of files to process
    ifstream ifs(indexfile.c_str());
    if(!ifs.is_open())
    {
        cout << "Index file " << indexfile << " could not be opened!" << endl;
        return false;
    }

    while(ifs >> filename)
    {
        filenames.push_back(filename);
    }
    ifs.close();

    return true;
}

int main (int argc, char** argv)
{
    if(!parseArgs(argc,argv)) return 0;

    float sum = colorW + normalW + curvatureW + ptplW;
    colorW /= sum;
    normalW /= sum;
    curvatureW /= sum;
    ptplW /= sum;

    // create output directories if necessary
    string unarystr = outputdir + "/unary";
    string pairwisestr = outputdir + "/pairwise";
    string labelstr = outputdir + "/labels";
    string clusteringstr = outputdir + "/clustering";

    cout << "Creating output directories in " << outputdir << endl;
    createOutputDirectories(clusteringstr, labelstr, pairwisestr, unarystr);

    vector<string> inputFiles;
    if(!loadIndexFile(inputFiles))
        return -1;

    // for augmentation
    CreateNRandomDeviations(inputFiles.size() * augmentations);

    vector<double> rollAngles;
    vector<double> pitchAngles;
    LoadAccelerometerPitchAndRoll(anglefile, rollAngles, pitchAngles);

    int nfiles = inputFiles.size();

    // track runtime
    std::vector<double> timing_prep(nfiles);
    std::vector<double> timing_seg(nfiles);
    std::vector<double> timing_feat(nfiles);

#ifdef NDEBUG
#pragma omp parallel for
#endif 
    for(int i=0; i < nfiles; ++i)
    {
        cout << "Processing " << inputFiles[i] << std::endl;

        // load pointcloud
        string cloudstr = cloudinputdir + "/" + inputFiles[i] + ".pcd";
        PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>());
        pcl::io::loadPCDFile(cloudstr, *cloud);

        // edit for paper to generate 2D images of the results
        pcl::PointCloud<pcl::PointXYZL> clusterIDsOrganized;
        pcl::copyPointCloud(*cloud, clusterIDsOrganized);
        // edit end

        // load ground truth
        string lblstr = lblcloudinputdir + "/" + inputFiles[i] + ".pcd";
        PointCloud<PointXYZL>::Ptr lblcloud (new PointCloud<PointXYZL>());
        pcl::io::loadPCDFile(lblstr, *lblcloud);

        pcl::IndicesPtr inliers(new std::vector <int>);

        if(outlierfilter)
        {                        
            // Statistical outlier removal
            // only get indices to remove later, after bilateral filter
            getInlierIndices(cloud, 50, 2.0, inliers);
        }
        else
        {
            inliers->resize(cloud->points.size());
            for(unsigned int j=0; j < cloud->points.size(); ++j)
            {
                inliers->at(j) = j;
            }
        }

        ptime time_prepstart(microsec_clock::local_time());

        // smoothing using bilateral filter
        if(bilateralfilter)
        {
            applyBilateralFilter(cloud, inliers, 10.0f, 0.05f, cloud);
        }

        ptime time_prepend(microsec_clock::local_time());
        time_duration duration_prep(time_prepend - time_prepstart);
        timing_prep[i] = duration_prep.total_milliseconds();

        if(outlierfilter)
        {
            // finally remove statistical outliers
            removeOutliers(cloud, inliers, cloud);
            removeOutliers(lblcloud, inliers, lblcloud);
        }

        Eigen::Matrix3f rotationMatrix;
        float realCamHeight = 0.0f;
        float camHeight = 0.0f;

        // try to fit plane to labeled floor plane first
        if(!CalculateHeightPitchAndRoll(lblcloud, realCamHeight, rotationMatrix))
        {
            int cloudidx = std::stoi(inputFiles[i])-1;

            if(cloudidx < 0 || cloudidx > (int)rollAngles.size())
            {
                std::cout << "No accelerometer angles for cloud " << inputFiles[i] << " found! Skip." <<endl;
                continue;
            }

            // plane fit failed, use accelerometer values for angles and fixed height as fallback
            GetAccelerometerPitchAndRoll(rollAngles[cloudidx], pitchAngles[cloudidx], rotationMatrix);
            realCamHeight = estimatedCameraHeight;
        }

        for(unsigned int j=0; j <= augmentations; ++j)
        {
            // for original version save outlier filtered point cloud (needed for evaluation)
            if(j == 0 && outlierfilter)
            {
                pcl::io::savePCDFileBinaryCompressed(lblcloudinputdir + "/" + inputFiles[i] + "_filtered.pcd", *lblcloud);
            }

            int idx = i*augmentations + j-1;
            double _seedResolution = seedResolution;
            double _colorImportance = colorImportance;
            double _normalImportance = normalImportance;
            double _spatialImportance = spatialImportance;
            double _mergingThreshold = mergingThreshold;
            double _colorW = colorW;
            double _normalW = normalW;
            double _curvatureW = curvatureW;
            double _ptplW = ptplW;

            if(j > 0)
            {
                // add random noise to create augmented versions
                _seedResolution += seedResolutionDev[idx];
                _colorImportance += colorImportanceDev[idx];
                _normalImportance += normalImportanceDev[idx];
                _spatialImportance += spatialImportanceDev[idx];
                _mergingThreshold += mergingThresholdDev[idx];
                _colorW += colorWDev[idx];
                _normalW += normalWDev[idx];
                _curvatureW += curvatureWDev[idx];
                _ptplW += ptplWDev[idx];
            }

            v4r::SupervoxelSegmentation<pcl::PointXYZRGB> segmenter(voxelResolution, _seedResolution,
                                                                _colorImportance, _normalImportance, _spatialImportance,
                                                                _mergingThreshold, _colorW, _normalW, _curvatureW, _ptplW, cie94, true);

            int nsegments = 0;

            if(j > 0)
            {
                // add random noise to camera position for augmentation
                double aug_yaw = augYawDev[idx];        // larger sigma for yaw!
                double aug_pitch = augPitchDev[idx];
                double aug_roll = augRollDev[idx];

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr augRotated (new pcl::PointCloud<pcl::PointXYZRGB>);
                createRotatedAugmentation(aug_roll, aug_pitch, aug_yaw, cloud, augRotated);

                // segment
                nsegments = segmenter.RunSegmentation(augRotated);

                camHeight = realCamHeight + augCamHeightDev[idx];
            }
            else
            {
                ptime time_segstart(microsec_clock::local_time());
                nsegments = segmenter.RunSegmentation(cloud);
                ptime time_segend(microsec_clock::local_time());

                time_duration duration_seg(time_segend - time_segstart);
                timing_seg[i] = duration_seg.total_milliseconds();

                camHeight = realCamHeight;
            }

            // get cluster ids for each point
            pcl::PointCloud<pcl::PointXYZL>::Ptr clusterIDs = segmenter.GetClusterIDPointcloud();

            pcl::PointCloud<pcl::PointXYZL>::Ptr v_clusterIDs(new pcl::PointCloud<pcl::PointXYZL>());
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr v_points(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::PointCloud<pcl::Normal>::Ptr v_normals(new pcl::PointCloud<pcl::Normal>());
            segmenter.GetVoxelizedResults(v_points, v_normals, v_clusterIDs);

            // storage for features
            std::vector< std::vector<double> > features;
            std::vector<std::vector<std::pair<double, int> > > sortedPointPlaneDistances;
            std::vector<std::vector<std::pair<double, int> > > sortedInversePointPlaneDistances;
            std::vector<std::vector<std::pair<double, int> > > sortedVerAngleDifferences;
            std::vector<std::vector<std::pair<double, int> > > sortedHorAngleDifferences;
            std::vector<std::vector<std::pair<double, int> > > sortedEuclideanDistances;

            // extract features
            v4r::EntangledForestFeatureExtraction featureExtractor;
            ptime time_featstart(microsec_clock::local_time());
            featureExtractor.setCameraExtrinsics(camHeight, rotationMatrix);
            featureExtractor.setInputCloud(v_points, v_normals, v_clusterIDs, nsegments);
            featureExtractor.extract();
            featureExtractor.getFeatures(features, sortedPointPlaneDistances, sortedInversePointPlaneDistances, sortedVerAngleDifferences,
                                         sortedHorAngleDifferences, sortedEuclideanDistances);
            ptime time_featend(microsec_clock::local_time());

            time_duration duration_feat(time_featend - time_featstart);
            timing_feat[i] = duration_feat.total_milliseconds();

            std::vector<int> clusterLabels;
            segmenter.GetLabelsForClusters(lblcloud, clusterLabels);

            if(j == 0)
            {
                // don't do it for augmented clouds

                // for classification later
                pcl::io::savePCDFileBinaryCompressed(clusteringstr + "/" + inputFiles[i] + ".pcd", *clusterIDs);

                // edit for paper to get organized results               

                for(unsigned int l=0; l<clusterIDsOrganized.points.size(); ++l)
                {
                    clusterIDsOrganized.at(l).label = std::numeric_limits<unsigned int>::max();
                }

                for(unsigned int m=0; m<inliers->size(); ++m)
                {
                    clusterIDsOrganized.at(inliers->at(m)).label = clusterIDs->at(m).label;
                }

                pcl::io::savePCDFileBinaryCompressed(clusteringstr + "/" + inputFiles[i]  + "_organized.pcd", clusterIDsOrganized);
                // end of edit
            }

            if(j == 0)
            {
                // save features and labels to files
                saveUnaryFeatures(unarystr + "/" + inputFiles[i], features);
                savePairwiseFeatures(pairwisestr + "/" + inputFiles[i] + "_euclid", sortedEuclideanDistances);
                savePairwiseFeatures(pairwisestr + "/" + inputFiles[i] + "_ptpl", sortedPointPlaneDistances);
                savePairwiseFeatures(pairwisestr + "/" + inputFiles[i] + "_iptpl", sortedInversePointPlaneDistances);
                savePairwiseFeatures(pairwisestr + "/" + inputFiles[i] + "_vangles", sortedVerAngleDifferences);
                savePairwiseFeatures(pairwisestr + "/" + inputFiles[i] + "_hangles", sortedHorAngleDifferences);
                saveLabels(labelstr + "/" + inputFiles[i], clusterLabels);
            }
            else
            {
                // save features and labels to files
                stringstream ss;
                ss << inputFiles[i] << "_" << j;
                saveUnaryFeatures(unarystr + "/" + ss.str(), features);
                savePairwiseFeatures(pairwisestr + "/" + ss.str() + "_euclid", sortedEuclideanDistances);
                savePairwiseFeatures(pairwisestr + "/" + ss.str() + "_ptpl", sortedPointPlaneDistances);
                savePairwiseFeatures(pairwisestr + "/" + ss.str() + "_iptpl", sortedInversePointPlaneDistances);
                savePairwiseFeatures(pairwisestr + "/" + ss.str() + "_vangles", sortedVerAngleDifferences);
                savePairwiseFeatures(pairwisestr + "/" + ss.str() + "_hangles", sortedHorAngleDifferences);
                saveLabels(labelstr + "/" + ss.str(), clusterLabels);
            }
        }

    }

    // save execution times
    ofstream prepfile(outputdir + "/timing_prep");
    for(unsigned int i=0; i<timing_prep.size(); ++i)
        prepfile << timing_prep[i] << std::endl;
    prepfile.close();

    ofstream segfile(outputdir + "/timing_seg");
    for(unsigned int i=0; i<timing_seg.size(); ++i)
        segfile << timing_seg[i] << std::endl;
    segfile.close();

    ofstream featfile(outputdir + "/timing_feat");
    for(unsigned int i=0; i<timing_feat.size(); ++i)
        featfile << timing_feat[i] << std::endl;
    featfile.close();

}
