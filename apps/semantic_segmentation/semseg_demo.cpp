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
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <ctime>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/random.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <opencv2/opencv.hpp>

#include <v4r/semantic_segmentation/supervoxel_segmentation.h>
#include <v4r/semantic_segmentation/entangled_forest.h>
#include <v4r/semantic_segmentation/entangled_feature_extraction.h>

namespace po = boost::program_options;
using namespace boost::posix_time;
using namespace std;

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

bool cie94;
bool bilateralfilter;

int ntrees;
int maxDepth;

float camHeight;
float camPitch;
float camRoll;

std::string inputfile;
std::string forestfile;
std::string colorfile;
std::string outputfile;

std::map<int,std::array<int, 3> > colorCode;

static bool parseArgs(int argc, char** argv)
{
    po::options_description genparam("General parameters");
    genparam.add_options()
            ("help,h","")
            ("input,i", po::value<std::string>(&inputfile), "Input point cloud")
            ("output,o", po::value<std::string>(&outputfile)->default_value("result.pcd"), "Output file for result")
            ("forest-file,f", po::value<string>(&forestfile), "Entangled forest classifier file" )
            ("colors,c", po::value<std::string>(&colorfile)->default_value("colors"), "Color code file for result pointclouds")
            ;

    po::options_description camparam("Camera parameters");
    camparam.add_options()
            ("cam-height", po::value<float>(&camHeight)->default_value(1.0), "Camera height above ground" )
            ("cam-pitch", po::value<float>(&camPitch)->default_value(0.0), "Camera pitch wrt ground plane (degrees)" )
            ("cam-roll", po::value<float>(&camRoll)->default_value(0.0), "Camera roll wrt ground plane (degrees)" )
            ;

    po::options_description prepparam("Preprocessing parameters");
    prepparam.add_options()
            ("bilfilter,b", po::value<bool>(&bilateralfilter)->default_value(true), "Use bilateral filter" )
            ;

    po::options_description segparam("Segmentation parameters");
    segparam.add_options()
            ("voxel-resolution,v", po::value<float>(&voxelResolution)->default_value(0.01), "Resolution of voxel grid" )
            ("seed-resolution,r", po::value<float>(&seedResolution)->default_value(0.3), "Resolution of supervoxel seeds" )
            ("color-importance,l", po::value<float>(&colorImportance)->default_value(0.6), "Weight of color features for clustering" )
            ("spatial-importance,s", po::value<float>(&spatialImportance)->default_value(0.3), "Weight of spatial features for clustering" )
            ("normal-importance,n", po::value<float>(&normalImportance)->default_value(1.0), "Weight of normal features for clustering" )
            ("merging-threshold,m", po::value<float>(&mergingThreshold)->default_value(0.06), "Max. dissimilarity measure to merge supervoxels" )
            ("merging-color", po::value<float>(&colorW)->default_value(1.0), "Merge weight for color" )
            ("merging-normals", po::value<float>(&normalW)->default_value(1.0), "Merge weight for normals" )
            ("merging-curvature", po::value<float>(&curvatureW)->default_value(0.0), "Merge weight for curvature" )
            ("merging-ptpl", po::value<float>(&ptplW)->default_value(1.0), "Merge weight for ptpl distance" )
            ("cie94", po::value<bool>(&cie94)->default_value(true), "Use CIE94 color distance" )
            ;

    po::options_description claparam("Classifier parameters");
    claparam.add_options()
            ("trees,t", po::value<int>(&ntrees)->default_value(-1), "Use n trees" )
            ("maxdepth,d", po::value<int>(&maxDepth)->default_value(-1), "Max. tree depth" )
            ;

    po::options_description all("");
    all.add(genparam).add(camparam).add(prepparam).add(segparam).add(claparam);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(all).run(), vm);

    po::notify(vm);

    if(vm.count("help") || !vm.count("forest-file") || !vm.count("input"))
    {
        std::cout << "General usage: classification_demo [options] -f forest-file -i input-file -o output-file -c color-file" << std::endl;
        std::cout << all;
        return false;
    }

    return true;
}

static void LoadColorCode()
{
    if(!colorfile.empty())
    {
        std::cout << "Load color code..." << std::endl;

        colorCode.clear();

        ifstream ifs(colorfile);
        int linecnt = 0;
        int elementcnt = 0;
        int value;

        std::array<int, 3> color;

        while (ifs >> value)
        {
            color[elementcnt++] = value;

            if (elementcnt == 3)
            {
                elementcnt = 0;
                colorCode[linecnt++] = color;
            }
        }

        ifs.close();
    }
    else
    {
        std::cout << "No color code given. RGB channels of result will be empty." << std::endl;
    }
}

static void applyBilateralFilter(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr inputcloud, double sigmaS, double sigmaR, pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputcloud)
{
    std::cout << "Apply bilateral filter..." << std::endl;
    pcl::FastBilateralFilterOMP<pcl::PointXYZRGB> bf;
    bf.setInputCloud (inputcloud);
    bf.setSigmaS(sigmaS);
    bf.setSigmaR(sigmaR);
    bf.filter(*outputcloud);
}

static void fillResultCloud(pcl::PointCloud<pcl::PointXYZL>::Ptr segmentation, std::vector<int>& result, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr resultCloud)
{
    std::cout << "Generate result point cloud..." << std::endl;

    // prepare result point cloud for display
    pcl::copyPointCloud(*segmentation, *resultCloud);

    std::array<int, 3> color;

    for(unsigned int i=0; i < segmentation->points.size(); ++i)
    {
        pcl::PointXYZRGBL &rgb_pt = resultCloud->points[i];

        unsigned int clusterlbl = segmentation->at(i).label;
        int resultlbl = 0;

        if(clusterlbl < result.size())
        {
            resultlbl = result[clusterlbl];
            // valid result
            color = colorCode[resultlbl];
        }
        else
        {
            // point has not been labeled, e.g. filtered out
            color = {0,0,0};
            rgb_pt.x = 0.0;
            rgb_pt.y = 0.0;
            rgb_pt.z = 0.0;
            rgb_pt.label = 0;
        }

        rgb_pt.r = color[0];
        rgb_pt.g = color[1];
        rgb_pt.b = color[2];
        rgb_pt.label = resultlbl;
    }
}

int main (int argc, char** argv)
{
    if(!parseArgs(argc,argv)) return 0;

    // merge weights of segmenter
    float sum = colorW + normalW + curvatureW + ptplW;
    colorW /= sum;
    normalW /= sum;
    curvatureW /= sum;
    ptplW /= sum;

    LoadColorCode();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::cout << "Load input point cloud..." << std::endl;
    if(pcl::io::loadPCDFile(inputfile.c_str(), *inputCloud) == -1)
    {
        PCL_ERROR("Couldn't read file %s.\n", inputfile.c_str());
        return -1;
    }

    // optionally smooth cloud with bilateral filter
    if(bilateralfilter)
    {
        applyBilateralFilter(inputCloud, 10, 0.05f, inputCloud);
    }

    // Initialize and run segmentation
    v4r::SupervoxelSegmentation<pcl::PointXYZRGB> segmenter(voxelResolution, seedResolution,
                                                       colorImportance, normalImportance, spatialImportance,
                                                       mergingThreshold, colorW, normalW, curvatureW, ptplW, cie94, true);

    std::cout << "Run segmentation..." << std::endl;
    // we do not provide normals and let the segmenter calculate them
    int nsegments = segmenter.RunSegmentation(inputCloud, NULL);

    // get segmentation result
    pcl::PointCloud<pcl::PointXYZL>::Ptr clusterIDs = segmenter.GetClusterIDPointcloud();

    // get voxelized segmentation result for feature extraction
    // classifier would also work with not voxelized data, but our segmenter only calculates normals on voxel level
    pcl::PointCloud<pcl::PointXYZL>::Ptr v_clusterIDs(new pcl::PointCloud<pcl::PointXYZL>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr v_points(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::Normal>::Ptr v_normals(new pcl::PointCloud<pcl::Normal>());
    segmenter.GetVoxelizedResults(v_points, v_normals, v_clusterIDs);

    std::cout << "Extract unary features..." << std::endl;
    // calculate unary features
    v4r::EntangledForestFeatureExtraction feat;
    feat.setInputCloud(v_points, v_normals, v_clusterIDs, nsegments);
    feat.setCameraExtrinsics(camHeight, DEG2RAD(camPitch), DEG2RAD(camRoll));
    feat.extract();

    // load classifier and run classification
    std::vector<int> result(0);                                    // storage for result (labelID per segment)
    v4r::EntangledForest f;                                        // forest object
    v4r::EntangledForestData d;                                    // data container

    std::cout << "Load classifier..." << std::endl;
    v4r::EntangledForest::LoadFromBinaryFile(forestfile, f);       // load forest
    feat.prepareClassification(&d);                                // init data container with unary features

    std::cout << "Classify..." << std::endl;
    f.Classify(&d, result, maxDepth, ntrees);                      // classify

    // generate results point cloud and save it
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr resultCloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
    fillResultCloud(clusterIDs, result, resultCloud);
    pcl::io::savePCDFileBinaryCompressed(outputfile.c_str(), *resultCloud);

    std::cout << "Result stored at " << outputfile << std::endl;
}
