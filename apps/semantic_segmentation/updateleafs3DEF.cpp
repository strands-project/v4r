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
#include <ctime>

#include <opencv2/core/core.hpp>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/program_options.hpp>

#include <v4r/semantic_segmentation/entangled_data.h>
#include <v4r/semantic_segmentation/entangled_forest.h>

using namespace std;
using namespace boost::posix_time;
namespace po = boost::program_options;


string inputfile;
string outputfile;

int depth;
string datadir;
string indexfile;
double updateWeight;

bool bagUniformly;

static bool parseArgs(int argc, char **argv)
{
    po::options_description forest("Options");
    forest.add_options()
            ("help,h", "")
            ("input,i", po::value<string>(&inputfile), "Input forest file")
            ("output,o", po::value<std::string>(&outputfile)->default_value("output.ef"), "Output forest file")
            ("trainingdata,t", po::value<std::string>(&datadir)->default_value("."),
             "Input directory of training data")
            ("indexfile,x", po::value<std::string>(&indexfile)->default_value("indextraining"),
             "Index file of training data")
            ("depth,d", po::value<int>(&depth)->default_value(100), "Depth level to update")
            ("updateweight,u", po::value<double>(&updateWeight)->default_value(1.0),
             "Weight of new distribution compared to old one")
            ("unibags,b", po::value<bool>(&bagUniformly)->default_value(false),
             "Try to uniformly sample training data");

    po::options_description all("");
    all.add(forest);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
            options(all).run(), vm);

    po::notify(vm);

    std::string usage = "General usage: updateleafs -i inputfile -o outputfile";

    if (vm.count("help"))
    {
        std::cout << usage << std::endl;
        std::cout << all;
        return false;
    }

    return true;
}

int main(int argc, char **argv)
{
    if (!parseArgs(argc, argv))
        return -1;

    cout << "Load forest " << inputfile << endl;
    v4r::EntangledForest f;
    v4r::EntangledForest::LoadFromBinaryFile(inputfile, f);

    cout << "Load training data from " << datadir << endl;
    v4r::EntangledForestData d;
    d.LoadTrainingData(datadir, indexfile);

    // update leaf nodes at certain depth and cut off deeper nodes
    f.updateLeafs(&d, depth, updateWeight, bagUniformly);

    cout << "DONE. Save new forest as " << outputfile << endl;
    f.SaveToBinaryFile(outputfile);
    cout << "DONE" << endl;
}
