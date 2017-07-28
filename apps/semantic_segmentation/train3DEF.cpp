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

#include <boost/program_options.hpp>

#include <v4r/semantic_segmentation/entangled_data.h>
#include <v4r/semantic_segmentation/entangled_forest.h>

using namespace std;
namespace po = boost::program_options;

int nTrees;
int maxDepth;
float bagging;
int nFeatures;
int nThresholds;
float minGain;
int minPoints;
string forestfile;

bool uniformBags;

string datadir;
string indexfile;
string namesfile;

static bool parseArgs(int argc, char** argv)
{
    po::options_description forest("Forest options");
    forest.add_options()
            ("help,h","")
            ("output,o", po::value<std::string>(&forestfile), "" )
            ("trees,t", po::value<int>(&nTrees)->default_value(60), "Number of trees")
            ("depth,d", po::value<int>(&maxDepth)->default_value(20), "Max. depth of trees (-1 = no limit)")
            ("bagging,b", po::value<float>(&bagging)->default_value(0.3), "Bagging ratio (0-1.0)")
            ("uniformbags,u", po::value<bool>(&uniformBags)->default_value(false), "Uniformly bag training data" )
            ("features,f", po::value<int>(&nFeatures)->default_value(1000), "Max. number of sampled settings per feature")
            ("thresholds,s", po::value<int>(&nThresholds)->default_value(100), "Max. number of sampled thresholds per feature")
            ("gain,g", po::value<float>(&minGain)->default_value(0.02), "Min. gain to split")
            ("points,p", po::value<int>(&minPoints)->default_value(5), "Min. points to split")
            ;

    po::options_description data("Data options");
    data.add_options()
            ("input,i", po::value<std::string>(&datadir), "Input directory of training data")
            ("indexfile,x", po::value<std::string>(&indexfile), "Index file for training data")
            ("namesfile,n", po::value<std::string>(&namesfile), "File containing list of label names")
            ;

    po::options_description all("");
    all.add(data).add(forest);

    po::options_description visible("");
    visible.add(data).add(forest);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(all).run(), vm);

    po::notify(vm);

    if(vm.count("help") || !vm.count("input") || !vm.count("output") || !vm.count("indexfile") || !vm.count("namesfile"))
    {
        std::cout << "General usage: train3DEF [-options] -o output-file -i trainingdata-dir -x index-file -n classname-file" << std::endl;
        std::cout << visible;
        return false;
    }

    return true;
}

int main (int argc, char** argv)
{
    if(!parseArgs(argc, argv))
        return -1;

    v4r::EntangledForestData d;

    d.LoadTrainingData(datadir, indexfile, namesfile);

    v4r::EntangledForest f(nTrees, maxDepth, bagging, nFeatures, nThresholds, minGain, minPoints);
    f.Train(&d, uniformBags);
    f.SaveToBinaryFile(forestfile);

    return 0;
}
