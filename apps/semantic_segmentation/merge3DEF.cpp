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

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <v4r/semantic_segmentation/entangled_data.h>
#include <v4r/semantic_segmentation/entangled_forest.h>

using namespace std;
using namespace boost::posix_time;
namespace po = boost::program_options;

std::vector<string> forestfiles;
string outputfile;

static bool parseArgs(int argc, char** argv)
{
    po::options_description forest("Options");
    forest.add_options()
            ("help,h","")
            ("forests,f", po::value<std::vector< string > >(&forestfiles)->multitoken(), "Forest files to merge" )
            ("output,o", po::value<std::string>(&outputfile)->default_value("merged.ef"), "" )
            ;

    po::options_description all("");
    all.add(forest);


    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(all).run(), vm);

    po::notify(vm);

    std::string usage = "General usage: mergeforests -f forest1 forest2 ... -o outputfile";

    if(forestfiles.size() < 2)
    {
        std::cout << "You have to list at least 2 forest files!" << std::endl;
        return false;
    }

    if(vm.count("help"))
    {
        std::cout << usage << std::endl;
        std::cout << all;
        return false;
    }

    return true;
}

int main (int argc, char** argv)
{
    if(!parseArgs(argc, argv))
        return -1;

    cout << "Load forest " << forestfiles[0] << endl;
    v4r::EntangledForest merged;
    v4r::EntangledForest::LoadFromBinaryFile(forestfiles[0], merged);

    for(unsigned int i=1; i < forestfiles.size(); ++i)
    {
        cout << "Merge with forest " << forestfiles[i] << endl;
        v4r::EntangledForest tomerge;
        v4r::EntangledForest::LoadFromBinaryFile(forestfiles[i], tomerge);
        merged.Merge(tomerge);
    }

    cout << "DONE. Save new forest as " << outputfile << endl;
    merged.SaveToBinaryFile(outputfile);
    cout << "DONE"<< endl;
}
