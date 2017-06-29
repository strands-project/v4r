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
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <ctime>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>

#include <boost/program_options.hpp>

#include <v4r/semantic_segmentation/supervoxel_segmentation.h>
#include <v4r/semantic_segmentation/entangled_forest.h>
#include <v4r/semantic_segmentation/entangled_feature_extraction.h>


namespace po = boost::program_options;
using namespace boost::posix_time;

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
bool outlierfilter;

bool singleframe;
bool integralNormals;

v4r::SupervoxelSegmentation<pcl::PointXYZRGB> *segmenter;

std::string inputfile;
std::string outputfile;

boost::mutex mtx;

bool segmentationDone = false;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr mergedCloud;

static bool parseArgs(int argc, char** argv)
{
    po::options_description sv("Supervoxel parameters");
    sv.add_options()
            ("help,h","")
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
            ("int", po::value<bool>(&integralNormals)->default_value(false), "Use integral normals")
            ("bilfilter,b", po::value<bool>(&bilateralfilter)->default_value(false), "Use bilateral filter" )
            ("outlierfilter,l", po::value<bool>(&outlierfilter)->default_value(false), "Use statistical outlier filter" )
            ("singleframe,f", po::value<bool>(&singleframe)->default_value(true), "Use single frame transform")
            ("output-file,o", po::value<std::string>(&outputfile)->default_value("supervoxels.pcd"), "")
            ;


    po::options_description hidden("Hidden parameters");
    hidden.add_options()
            ("pointcloudfile,i", po::value<std::string>(&inputfile), "")
            ;

    po::positional_options_description pos;
    pos.add("pointcloudfile",-1);

    po::options_description all("");
    all.add(sv).add(hidden);

    po::options_description visible("");
    visible.add(sv);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(all).positional(pos).run(), vm);

    po::notify(vm);

    std::string usage = "General usage: supervoxels pointcloudfile";

    if(vm.count("help"))
    {
        std::cout << usage << std::endl;
        std::cout << visible;
        return false;
    }
    if(!vm.count("pointcloudfile"))
    {
        inputfile = "";
    }

    return true;
}


static void getInlierIndices(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, int meanK, double stddevMulThresh, pcl::IndicesPtr inlier_indices)
{
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
    for(unsigned int i=0; i<inlier_indices->size(); ++i)
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

static void applyBilateralFilter(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr inputcloud, pcl::IndicesConstPtr indices, double sigmaS, double sigmaR, pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputcloud)
{
    pcl::FastBilateralFilterOMP<pcl::PointXYZRGB> bf;
    bf.setInputCloud (inputcloud);
    bf.setIndices(indices);
    bf.setSigmaS(sigmaS);
    bf.setSigmaR(sigmaR);
    bf.filter(*outputcloud);
}

static void removeOutliers(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr inputcloud, pcl::IndicesConstPtr indices, pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputcloud)
{
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(inputcloud);
    extract.setIndices(indices);
    extract.setNegative(false);
    extract.filter(*outputcloud);
}

static void segmentPointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals = NULL)
{
    ptime time_start(microsec_clock::local_time());

    segmenter->RunSegmentation(cloud, normals);

    ptime time_seg(microsec_clock::local_time());
    time_duration duration1(time_seg - time_start);
    std::cout << "Supervoxels incl. merging: " << duration1.total_milliseconds() << " ms" << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered = segmenter->getColoredClusterPointCloud();

    {
        boost::lock_guard<boost::mutex> lock(mtx);
        pcl::copyPointCloud(*cloud, *pointcloud);
        pcl::copyPointCloud(*clustered, *mergedCloud);
        segmentationDone = true;
    }
}

static void preprocessAndSegment (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{
    ptime time_start(microsec_clock::local_time());

    pcl::IndicesPtr inlier_indices(new std::vector <int>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr prepcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::copyPointCloud(*cloud, *prepcloud);

    if(outlierfilter)
    {
        // Statistical outlier removal
        getInlierIndices(prepcloud, 50, 2.0, inlier_indices);
    }

    if(bilateralfilter)
    {
        // Bilateral filter (does not change point arrangement)
        applyBilateralFilter(prepcloud, inlier_indices, 10, 0.05f, prepcloud);  // was 0.1!
    }

    if(outlierfilter)
    {
        removeOutliers(prepcloud, inlier_indices, prepcloud);
    }

    ptime time_preprocessing(microsec_clock::local_time());
    time_duration duration1(time_preprocessing - time_start);
    std::cout << "Preprocessing time: " << duration1.total_milliseconds() << " ms" << std::endl;

    segmentPointcloud(prepcloud);
}

static bool normalsAvailable(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud)
{
    // returns true if at least one normal is available (> 0), otherwise false

    float eps = 1e-10;

    for(unsigned int k=0; k<cloud->size(); ++k)
    {
        pcl::PointXYZRGBNormal &pt =  cloud->at(k);
        if(pcl::isFinite(pt))
        {
            if(fabs(pt.normal_x) > eps || fabs(pt.normal_y) > eps || fabs(pt.normal_z) > eps)
            {
                return true;
            }
        }
    }

    return false;
}

int main (int argc, char** argv)
{
    if(!parseArgs(argc,argv)) return 0;

    float sum = colorW + normalW + curvatureW + ptplW;
    colorW /= sum;
    normalW /= sum;
    curvatureW /= sum;
    ptplW /= sum;

    pointcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >();
    mergedCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >();

    pcl::Grabber* interface;

    segmenter = new v4r::SupervoxelSegmentation<pcl::PointXYZRGB> (voxelResolution, seedResolution,
                                                                             colorImportance, normalImportance, spatialImportance,
                                                                             mergingThreshold, colorW, normalW, curvatureW, ptplW, cie94, singleframe);


    if(!inputfile.empty() && inputfile.length() > 4)
    {                        
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudwnormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

        std::string extension = inputfile.substr(inputfile.length() - 4, 4);

        if(!extension.compare(".pcd"))
        {
            // PCD file            
            if(pcl::io::loadPCDFile(inputfile.c_str(), *cloudwnormals) == -1)
            {
                PCL_ERROR("Couldn't read with normals %s.\n", inputfile.c_str());
                return -1;
            }

            // check if normals could be read
            if(normalsAvailable(cloudwnormals))
            {
                pcl::copyPointCloud(*cloudwnormals, *cloud);
                pcl::copyPointCloud(*cloudwnormals, *normals);

                segmentPointcloud(cloud, normals);
            }
            else
            {
                if(pcl::io::loadPCDFile(inputfile.c_str(), *cloud) == -1)
                {
                    PCL_ERROR("Couldn't read file %s.\n", inputfile.c_str());
                    return -1;
                }

                preprocessAndSegment(cloud);
            }
        }
        else if(!extension.compare(".ply"))
        {
            // PLY file                        
            pcl::PLYReader ply;
            ply.read(inputfile, *cloudwnormals);

            pcl::copyPointCloud(*cloudwnormals, *cloud);
            pcl::copyPointCloud(*cloudwnormals, *normals);

            segmentPointcloud(cloud, normals);
        }
        else
        {
            std::cout << "Invalid file extension!" << std::endl;
            return -1;
        }
    }
    else
    {
        interface = new pcl::OpenNIGrabber();
        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> f =
                boost::bind (preprocessAndSegment, _1);

        interface->registerCallback (f);

        interface->start ();
    }

    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    int v1, v2;

    while (!viewer.wasStopped ())
    {        
        {
            boost::unique_lock<boost::mutex> lock(mtx);

            if(segmentationDone)
            {
                segmentationDone = false;

                // to visualize rgb input cloud
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(pointcloud);

                if(!viewer.updatePointCloud(pointcloud, handler, "original"))
                {
                    viewer.createViewPort(0,0,0.5,1, v1);
                    viewer.setBackgroundColor(1.0, 1.0, 1.0, v1);
                    viewer.addPointCloud (pointcloud, handler, "original", v1);
                }

                if(!viewer.updatePointCloud (mergedCloud, "merged dense"))
                {                    
                    viewer.createViewPort(0.5,0,1,1, v2);
                    viewer.setBackgroundColor(1.0, 1.0, 1.0, v2);
                    viewer.addPointCloud (mergedCloud, "merged dense",v2);
                }
            }
        }

        viewer.spinOnce (100);
    }

    if(inputfile.empty())
    {
        interface->stop();
    }

    delete segmenter;
}
