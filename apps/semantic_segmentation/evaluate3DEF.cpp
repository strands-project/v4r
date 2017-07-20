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

#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>

#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <v4r/semantic_segmentation/entangled_forest.h>
#include <v4r/semantic_segmentation/entangled_data.h>

using namespace boost::posix_time;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;

string forestfile;
int maxDepth;
int nTrees;
string datadir;
string indexfile;
string outputdir;
string labeldir;
string colorfile;
bool filtered;

std::map<int,std::array<int, 3> > colorCode;
bool savePointclouds;

static bool parseArgs(int argc, char** argv)
{
    po::options_description general("General options");
    general.add_options()
            ("help", "show help message")
            ("forest,f", po::value<std::string>(&forestfile)->default_value("forest.ef"), "Stored forest file" )
            ("depth,d", po::value<int>(&maxDepth)->default_value(0), "Max. depth to expand (0 = no limit)")
            ("trees,t", po::value<int>(&nTrees)->default_value(0), "Number of trees to use (0 = use all)")
            ("input,i", po::value<std::string>(&datadir)->default_value("."), "Input directory of test data")
            ("indexfile,x", po::value<std::string>(&indexfile)->default_value("indextest"), "Index file for test data")
            ("label,l", po::value<std::string>(&labeldir)->default_value("."), "Input directory of groundtruth pointclouds")
            ("colors,c", po::value<std::string>(&colorfile)->default_value("colors"), "Color code file for result pointclouds")
            ("output,o", po::value<std::string>(&outputdir)->default_value("evaluation"), "Output directory for evaluation data")
            ("savepc,s", po::value<bool>(&savePointclouds)->default_value(false), "Store result pointclouds")
            ;

    po::options_description all("");
    all.add(general);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(all).run(), vm);

    po::notify(vm);

    std::string usage = "General usage: evaluateentangled [options]";

    if(vm.count("help"))
    {
        std::cout << usage << std::endl;
        std::cout << all;
        return false;
    }    

    return true;
}

static void LoadColorCode()
{
    colorCode.clear();

    ifstream ifs(colorfile);
    int linecnt = 0;
    int elementcnt = 0;
    int value;

    std::array<int,3> color;

    while(ifs >> value)
    {
        color[elementcnt++] = value;

        if(elementcnt == 3)
        {
            elementcnt = 0;
            colorCode[linecnt++] = color;
        }
    }

    ifs.close();
}

static void EvaluateResult(v4r::EntangledForestData* data, string filename, std::vector<int>& result, std::vector<std::vector<int> >& confusionMatrix, pcl::PointCloud<pcl::PointXYZRGB>::Ptr resultPointcloud = NULL)
{  
    std::map<int, int>& labelMap = data->GetLabelMap();

    // initialize confusion matrix
    int nlabels = labelMap.size();
    confusionMatrix.resize(nlabels, std::vector<int>(nlabels, 0));

    // load groundtruth
    pcl::PointCloud<pcl::PointXYZL> gt;
    pcl::io::loadPCDFile(labeldir + "/" + filename + ".pcd", gt);

    // load segmentation result
    pcl::PointCloud<pcl::PointXYZL> clustering;
    pcl::io::loadPCDFile(datadir + "/clustering/" + filename + "_organized.pcd", clustering);

    if(resultPointcloud)
    {
        pcl::copyPointCloud(clustering, *resultPointcloud);
    }

    for(unsigned int i=0; i < clustering.points.size(); ++i)
    {
        unsigned int clusterlbl = clustering.at(i).label;
        int resultlbl = 0;
        unsigned int gtlbl = 0;

        if(clusterlbl < result.size())
        {
            resultlbl = result[clusterlbl];
            gtlbl = gt.at(i).label;

            if(gtlbl > 0 && resultlbl > 0)
            {
                confusionMatrix[labelMap[gtlbl]][labelMap[resultlbl]]++;
            }
        }

        if(resultPointcloud)
        {
            pcl::PointXYZRGB &rgb_pt = resultPointcloud->points[i];
            std::array<int, 3> color;

            if(resultlbl > 0)
            {
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
            }

            rgb_pt.r = color[0];
            rgb_pt.g = color[1];
            rgb_pt.b = color[2];
        }
    }
}


static void ConvertPCLCloud2Image (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud, cv::Mat_<cv::Vec3b> &image)
{
    unsigned pcWidth = pcl_cloud->width;
    unsigned pcHeight = pcl_cloud->height;
    unsigned position = 0;

    image = cv::Mat_<cv::Vec3b> (pcHeight, pcWidth);

    for (unsigned row = 0; row < pcHeight; row++)
    {
        for (unsigned col = 0; col < pcWidth; col++)
        {
            cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
            position = row * pcWidth + col;
            const pcl::PointXYZRGB &pt = pcl_cloud->points[position];

            if(pt.z < 0.01)
            {
                // point has not been labeled, mark as white in image
                cvp[0] = 255;
                cvp[1] = 255;
                cvp[2] = 255;
            }
            else
            {
                cvp[0] = pt.b;
                cvp[1] = pt.g;
                cvp[2] = pt.r;
            }
        }
    }
}


int main (int argc, char** argv)
{
    if(!parseArgs(argc, argv))
        return 0;

    std::cout << "Load classifier " << forestfile << endl;

    v4r::EntangledForest f;
    v4r::EntangledForest::LoadFromBinaryFile(forestfile, f);

    std::vector<string> filenames;
    string filename;

    ifstream ifs(indexfile.c_str());
    while(ifs >> filename)
    {
        filenames.push_back(filename);
    }
    ifs.close();

    // create output dir
    fs::path p(datadir + "/" + outputdir);

    if(!fs::exists(p))
        fs::create_directories(p);

    // store execution times for statistics
    std::vector<int> executionTimes;

    LoadColorCode();

    for(unsigned int i=0; i<filenames.size(); ++i)
    {
        filename = filenames[i];        

        cout << "Loading test data " << filename << endl;

        v4r::EntangledForestData d;
        if(!d.LoadTestData(datadir, filename))
        {
            cout << "File " << filename << " not available. Skip." << endl;
            continue;
        }       

        // classify
        std::vector<int> result;

        cout << "Classify..." << endl;
        ptime time_start(microsec_clock::local_time());        
        f.Classify(&d, result, maxDepth, nTrees);
        ptime time_end(microsec_clock::local_time());
        time_duration duration(time_end - time_start);

        std::cout << "Execution time: " << duration.total_milliseconds() << " ms" << std::endl;
        executionTimes.push_back(duration.total_milliseconds());

        std::vector<std::vector<int> > confusionMatrix;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr resultpc = NULL;

        if(savePointclouds)
        {
            resultpc.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        }

        EvaluateResult(&d, filename, result, confusionMatrix, resultpc);

        string outputfile = datadir + "/" + outputdir + "/" + filename;

        ofstream ofs(outputfile.c_str());

        for(unsigned int n=0; n<confusionMatrix.size(); ++n)
        {            
            for(unsigned int j=0; j<confusionMatrix.size(); ++j)
                ofs << confusionMatrix[n][j] << " ";

            ofs << endl;
        }

        ofs.close();

        if(savePointclouds)
        {
            pcl::io::savePCDFileBinaryCompressed(outputfile + ".pcd", *resultpc);
            cv::Mat_<cv::Vec3b> resultimage;
            ConvertPCLCloud2Image(resultpc, resultimage);
            cv::imwrite(outputfile + ".png", resultimage);
        }

    }

    string exectimes = datadir + "/" + outputdir + "/executiontimes";

    ofstream timing(exectimes.c_str());

    for(unsigned int i=0; i < executionTimes.size(); ++i)
        timing << executionTimes[i] << endl;

    timing.close();

}
