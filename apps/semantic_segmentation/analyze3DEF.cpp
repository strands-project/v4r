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
#include <ostream>
#include <vector>
#include <array>
#include <map>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <v4r/semantic_segmentation/entangled_data.h>
#include <v4r/semantic_segmentation/entangled_forest.h>
#include <v4r/semantic_segmentation/entangled_definitions.h>

using namespace std;
namespace po = boost::program_options;

string outputdir;
string fileindex;
string forestfile;
int maxDepth;

static bool parseArgs(int argc, char** argv)
{
    po::options_description forest("General options");
    forest.add_options()
            ("help,h","")
            ("forest,f", po::value<std::string>(&forestfile)->default_value("forest.ef"), "Stored forest file" )
            ("output-dir,o", po::value<std::string>(&outputdir)->default_value("featureanalysis"), "Output directory" )
            ;

    po::options_description visible("");
    visible.add(forest);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(forest).run(), vm);

    po::notify(vm);

    std::string usage = "General usage: analyzeforestCluster -f forest-file -o output-directory";

    if(vm.count("help"))
    {
        std::cout << usage << std::endl;
        std::cout << visible;
        return false;
    }

    return true;
}

static void CreateTopNHistograms(int nrOfLabels, std::vector< std::vector<v4r::EntangledForestTopNFeature* > >& pairwisetopn, std::vector< std::vector<v4r::EntangledForestInverseTopNFeature* > >& inversetopn,
                          std::vector<std::vector<unsigned int> >& pairwisetopnhisthor, std::vector<std::vector<unsigned int> >& pairwisetopnhistver, std::vector<std::vector<unsigned int> >& inversetopnhist)
{
    pairwisetopnhisthor.clear();
    pairwisetopnhistver.clear();
    inversetopnhist.clear();

    int maxdepth = pairwisetopn.size();
    pairwisetopnhisthor.assign(maxdepth, std::vector<unsigned int>(nrOfLabels, 0));
    pairwisetopnhistver.assign(maxdepth, std::vector<unsigned int>(nrOfLabels, 0));
    inversetopnhist.assign(maxdepth, std::vector<unsigned int>(nrOfLabels, 0));

    for(int d=0; d<maxdepth; ++d)
    {
        for(unsigned int i=0; i<pairwisetopn[d].size(); ++i)
        {
            v4r::EntangledForestTopNFeature* f =  pairwisetopn[d][i];
            int lbl = f->GetLabel();
            if(f->IsHorizontal())
            {
                pairwisetopnhisthor[d][lbl]++;
            }
            else
            {
                pairwisetopnhistver[d][lbl]++;
            }
        }

        for(unsigned int i=0; i<inversetopn[d].size(); ++i)
        {
            v4r::EntangledForestInverseTopNFeature* f =  inversetopn[d][i];
            int lbl = f->GetLabel();
            inversetopnhist[d][lbl]++;
        }
    }
}

static void CreateAngleAndDistLists(std::vector< std::vector<v4r::EntangledForestTopNFeature* > >& pairwisetopn, std::vector< std::vector<v4r::EntangledForestInverseTopNFeature* > >& inversetopn,
                             std::vector<std::vector<double> >& meananglehor, std::vector<std::vector<double> >& meananglever, std::vector<std::vector<double> >& meanangleinv,
                             std::vector<std::vector<double> >& meandisthor, std::vector<std::vector<double> >& meandistver, std::vector<std::vector<double> >& meandistinv,
                             std::vector<std::vector<double> >& anglecorrhor, std::vector<std::vector<double> >& anglecorrver, std::vector<std::vector<double> >& anglecorrinv,
                             std::vector<std::vector<double> >& distcorrhor, std::vector<std::vector<double> >& distcorrver, std::vector<std::vector<double> >& distcorrinv)
{
    meananglehor.clear();
    meananglever.clear();
    meanangleinv.clear();
    meandisthor.clear();
    meandistver.clear();
    meandistinv.clear();
    anglecorrhor.clear();
    anglecorrver.clear();
    anglecorrinv.clear();
    distcorrhor.clear();
    distcorrver.clear();
    distcorrinv.clear();

    int maxdepth = pairwisetopn.size();
    meananglehor.assign(maxdepth, std::vector<double>());
    meananglever.assign(maxdepth, std::vector<double>());
    meanangleinv.assign(maxdepth, std::vector<double>());
    meandisthor.assign(maxdepth, std::vector<double>());
    meandistver.assign(maxdepth, std::vector<double>());
    meandistinv.assign(maxdepth, std::vector<double>());
    anglecorrhor.assign(maxdepth, std::vector<double>());
    anglecorrver.assign(maxdepth, std::vector<double>());
    anglecorrinv.assign(maxdepth, std::vector<double>());
    distcorrhor.assign(maxdepth, std::vector<double>());
    distcorrver.assign(maxdepth, std::vector<double>());
    distcorrinv.assign(maxdepth, std::vector<double>());

    std::vector<double> parameters;

    for(int d=0; d<maxdepth; ++d)
    {
        for(unsigned int i=0; i<pairwisetopn[d].size(); ++i)
        {
            v4r::EntangledForestTopNFeature* f = pairwisetopn[d][i];
            parameters = f->GetGeometryParameters();
            if(parameters[1]-parameters[0] < 6.28f)
            {
                if(f->IsHorizontal())
                {
                    meananglehor[d].push_back((parameters[0] + parameters[1]) / 2.0);
                    anglecorrhor[d].push_back(parameters[1]-parameters[0]);
                }
                else
                {
                    meananglever[d].push_back((parameters[0] + parameters[1]) / 2.0);
                    anglecorrver[d].push_back(parameters[1]-parameters[0]);
                }
            }
            if(parameters[3]-parameters[2] < 200.0f)
            {
                if(f->IsHorizontal())
                {
                    meandisthor[d].push_back((parameters[2] + parameters[3]) / 2.0);
                    distcorrhor[d].push_back(parameters[3]-parameters[2]);
                }
                else
                {
                    meandistver[d].push_back((parameters[2] + parameters[3]) / 2.0);
                    distcorrver[d].push_back(parameters[3]-parameters[2]);
                }
            }
        }

        for(unsigned int i=0; i<inversetopn[d].size(); ++i)
        {
            v4r::EntangledForestInverseTopNFeature* f =  inversetopn[d][i];
            parameters = f->GetGeometryParameters();
            if(parameters[1]-parameters[0] < 6.28f)
            {
                meanangleinv[d].push_back((parameters[0] + parameters[1]) / 2.0);
                anglecorrinv[d].push_back(parameters[1]-parameters[0]);
            }
            if(parameters[3]-parameters[2] < 200.0f)
            {
                meandistinv[d].push_back((parameters[2] + parameters[3]) / 2.0);
                distcorrinv[d].push_back(parameters[3]-parameters[2]);
            }
        }
    }
}


static void SaveLabel2LabelRelations(v4r::EntangledForest* f, string filename)
{
    ofstream l2l(filename);

    for(int t=0; t < f->GetNrOfTrees(); ++t)
    {
        v4r::EntangledForestTree* tree = f->GetTree(t);

        for(int n=0; n < tree->GetNrOfNodes(); ++n)
        {
            v4r::EntangledForestNode* node = tree->GetNode(n);
            if(node->IsSplitNode())
            {
                int depth = node->GetDepth();
                v4r::EntangledForestSplitFeature *feature = node->GetSplitFeature();
                string featurename = feature->GetName();
                string feattype;
                double angle;
                double diff;
                int targetlbl;
                int splitlbl;

                if(featurename.find("Pairwise top n") != string::npos)
                {
                    v4r::EntangledForestTopNFeature* topn = (v4r::EntangledForestTopNFeature*) feature;
                    feattype = topn->IsHorizontal() ? "hor" : "ver";

                    // skip don't care for angle
                    if(topn->GetGeometryParameters()[1] > 6.2)
                        continue;

                    angle = (topn->GetGeometryParameters()[0]+topn->GetGeometryParameters()[1]) / 2.0;
                    diff = (topn->GetGeometryParameters()[2]+topn->GetGeometryParameters()[3]) / 2.0;
                    targetlbl = topn->GetLabel()+1;
                }
                else if(featurename.find("inverse") != string::npos)
                {
                    v4r::EntangledForestInverseTopNFeature* inverse = (v4r::EntangledForestInverseTopNFeature*) feature;
                    feattype = "inv";

                    // skip don't care for angle
                    if(inverse->GetGeometryParameters()[1] > 6.2)
                        continue;

                    angle = (inverse->GetGeometryParameters()[0]+inverse->GetGeometryParameters()[1]) / 2.0;
                    diff = (inverse->GetGeometryParameters()[2]+inverse->GetGeometryParameters()[3]) / 2.0;
                    targetlbl = inverse->GetLabel()+1;
                }
                else
                {
                    continue;
                }

                // figure out split label
//                Node* left = tree->GetNode(node->GetLeftChildIdx());
                v4r::EntangledForestNode* right = tree->GetNode(node->GetRightChildIdx());

                vector<double>& parentdist = node->GetLabelDistribution();
                vector<double>& rightdist = right->GetLabelDistribution();
                vector<double> entropydiff(parentdist.size());

                for(unsigned int l=0; l < parentdist.size(); ++l)
                {
//                    double parentbin = -log2(parentdist[l]+std::numeric_limits<double>::min())*parentdist[l];
//                    double rightbin = -log2(rightdist[l]+std::numeric_limits<double>::min())*rightdist[l];
                    entropydiff[l] = rightdist[l]-parentdist[l];//rightbin-parentbin;
                }

                splitlbl = std::distance(entropydiff.begin(), std::max_element(entropydiff.begin(), entropydiff.end()))+1;

                l2l << setw(2) << depth << " " << feattype << " " << setw(2) << targetlbl << " " << setw(2) << splitlbl << " " << angle << " " << diff << std::endl;
            }
        }
    }

    l2l.close();
}


int main (int argc, char** argv)
{
    if(!parseArgs(argc, argv))
        return -1;

    v4r::EntangledForest f;

    LOG_INFO("Load classifier...");
    v4r::EntangledForest::LoadFromBinaryFile(forestfile, f);
    LOG_INFO("DONE.");

    // first, find all used features
    set<string> featurelist;
    int maxdepth = 0;

    for(int tidx=0; tidx < f.GetNrOfTrees(); ++tidx)
    {
        v4r::EntangledForestTree* t = f.GetTree(tidx);

        for(int nidx=0; nidx < t->GetNrOfNodes(); ++nidx)
        {
            v4r::EntangledForestNode* n = t->GetNode(nidx);
            if(n->IsSplitNode())
            {
                string name = n->GetSplitFeature()->GetName();
                featurelist.insert(name);

                int depth = n->GetDepth();
                if(depth > maxdepth)
                {
                    maxdepth = depth;
                }
            }
        }
    }

    // initialize histograms
    vector<vector<v4r::EntangledForestUnaryFeature*> > unaryfeatures(maxdepth+1);
    vector<vector<v4r::EntangledForestClusterExistsFeature*> > clusterexistsfeatures(maxdepth+1);
    vector<vector<v4r::EntangledForestTopNFeature*> > topnfeatures(maxdepth+1);
    vector<vector<v4r::EntangledForestInverseTopNFeature*> > itopnfeatures(maxdepth+1);
    vector<vector<v4r::EntangledForestCommonAncestorFeature*> > ancestorfeatures(maxdepth+1);
    vector<vector<v4r::EntangledForestNodeDescendantFeature*> > descendantfeatures(maxdepth+1);

    vector<vector<double> > topnhist(maxdepth+1, vector<double>(f.GetNrOfLabels(),0.0));
    vector<map<string, double> > feathist(maxdepth+1);

    for(int d=0; d<=maxdepth; ++d)
    {
        for(auto feat=featurelist.begin(); feat != featurelist.end(); ++feat)
        {
            feathist[d][*feat] = 0.0;
        }
    }

    // now create feature and label histograms
    for(int tidx=0; tidx < f.GetNrOfTrees(); ++tidx)
    {
        v4r::EntangledForestTree* t = f.GetTree(tidx);

        for(int nidx=0; nidx < t->GetNrOfNodes(); ++nidx)
        {
            v4r::EntangledForestNode* n = t->GetNode(nidx);
            if(n->IsSplitNode())
            {
                v4r::EntangledForestSplitFeature* feature = n->GetSplitFeature();
                string featurename = feature->GetName();
                int depth = n->GetDepth();

                if(featurename.find("Unary") != string::npos)
                {
                    unaryfeatures[depth].push_back((v4r::EntangledForestUnaryFeature*) feature);
                }
                else if(featurename.find("Pairwise feature") != string::npos)
                {
                    clusterexistsfeatures[depth].push_back((v4r::EntangledForestClusterExistsFeature*) feature);
                }
                else if(featurename.find("Pairwise top n") != string::npos)
                {
                    topnfeatures[depth].push_back((v4r::EntangledForestTopNFeature*) feature);
                }
                else if(featurename.find("inverse") != string::npos)
                {
                    itopnfeatures[depth].push_back((v4r::EntangledForestInverseTopNFeature*) feature);
                }
                else if(featurename.find("ancestor") != string::npos)
                {
                    ancestorfeatures[depth].push_back((v4r::EntangledForestCommonAncestorFeature*) feature);
                }
                else if(featurename.find("descendant") != string::npos)
                {
                    descendantfeatures[depth].push_back((v4r::EntangledForestNodeDescendantFeature*) feature);
                }


                feathist[depth][featurename]++;

                if(featurename.find("top n") != string::npos)
                {
                    unsigned int label = 0;

                    // inverse or normal top n?
                    if(featurename.find("inverse") != string::npos)
                    {
                        v4r::EntangledForestInverseTopNFeature* p = (v4r::EntangledForestInverseTopNFeature*)n->GetSplitFeature();
                        label = p->GetLabel();
                    }
                    else
                    {
                        v4r::EntangledForestTopNFeature* p = (v4r::EntangledForestTopNFeature*)n->GetSplitFeature();
                        label = p->GetLabel();
                    }

                    topnhist[depth][label]++;
                }
            }
        }
    }

    vector<vector<unsigned int> > topnhisthor;
    vector<vector<unsigned int> > topnhistver;
    vector<vector<unsigned int> > inversehist;

    vector<vector<double> > meanangleshor;
    vector<vector<double> > meananglesver;
    vector<vector<double> > meananglesinv;
    vector<vector<double> > meandisthor;
    vector<vector<double> > meandistver;
    vector<vector<double> > meandistinv;

    vector<vector<double> > anglecorrhor;
    vector<vector<double> > anglecorrver;
    vector<vector<double> > anglecorrinv;
    vector<vector<double> > distcorrhor;
    vector<vector<double> > distcorrver;
    vector<vector<double> > distcorrinv;

    CreateTopNHistograms(f.GetNrOfLabels(), topnfeatures, itopnfeatures, topnhisthor, topnhistver, inversehist);
    CreateAngleAndDistLists(topnfeatures, itopnfeatures, meanangleshor, meananglesver, meananglesinv, meandisthor, meandistver, meandistinv,
                            anglecorrhor, anglecorrver, anglecorrinv, distcorrhor, distcorrver, distcorrinv);

    boost::filesystem::create_directories(boost::filesystem::path(outputdir));

    // save unnormalized feature histogram /////

    ofstream ofsfeatures(outputdir + "/featurelist");

    // save list of features
    for(auto feat=featurelist.begin(); feat != featurelist.end(); ++feat)
    {
        ofsfeatures << *feat << std::endl;
    }

    ofsfeatures.close();

    ofstream ofs(outputdir + "/featurehist");

    for(int d=0; d < maxdepth; ++d)
    {
        for(auto feat=feathist[d].begin(); feat != feathist[d].end(); ++feat)
        {
            ofs << setw(13) << feat->second;
        }

        ofs << endl;
    }
    ofs.close();

    ////////////////////////////////////////////

    // save unnormalized top N label histograms and mean angles and dists
    ofstream topnhorfile(outputdir+"/topnhisthor");
    ofstream topnverfile(outputdir+"/topnhistver");
    ofstream topninvfile(outputdir+"/topnhistinv");
    ofstream meanangleshorfile(outputdir+"/angleshor");
    ofstream meananglesverfile(outputdir+"/anglesver");
    ofstream meananglesinvfile(outputdir+"/anglesinv");
    ofstream meandisthorfile(outputdir+"/disthor");
    ofstream meandistverfile(outputdir+"/distver");
    ofstream meandistinvfile(outputdir+"/distinv");
    ofstream anglecorrhorfile(outputdir+"/anglescorrhor");
    ofstream anglecorrverfile(outputdir+"/anglescorrver");
    ofstream anglecorrinvfile(outputdir+"/anglescorrinv");
    ofstream distcorrhorfile(outputdir+"/distcorrhor");
    ofstream distcorrverfile(outputdir+"/distcorrver");
    ofstream distcorrinvfile(outputdir+"/distcorrinv");

    for(int d=0; d < maxdepth; ++d)
    {
        for(unsigned int i=0; i < topnhisthor[d].size(); ++i)
        {
            topnhorfile << setw(13) << topnhisthor[d][i];
            topnverfile << setw(13) << topnhistver[d][i];
            topninvfile << setw(13) << inversehist[d][i];
        }

        for(unsigned int i=0; i < meanangleshor[d].size(); ++i)
            meanangleshorfile << setw(13) << meanangleshor[d][i];
        for(unsigned int i=0; i < meananglesver[d].size(); ++i)
            meananglesverfile << setw(13) << meananglesver[d][i];
        for(unsigned int i=0; i < meananglesinv[d].size(); ++i)
            meananglesinvfile << setw(13) << meananglesinv[d][i];
        for(unsigned int i=0; i < meandisthor[d].size(); ++i)
            meandisthorfile << setw(13) << meandisthor[d][i];
        for(unsigned int i=0; i < meandistver[d].size(); ++i)
            meandistverfile << setw(13) << meandistver[d][i];
        for(unsigned int i=0; i < meandistinv[d].size(); ++i)
            meandistinvfile << setw(13) << meandistinv[d][i];
        for(unsigned int i=0; i < anglecorrhor[d].size(); ++i)
            anglecorrhorfile << setw(13) << anglecorrhor[d][i];
        for(unsigned int i=0; i < anglecorrver[d].size(); ++i)
            anglecorrverfile << setw(13) << anglecorrver[d][i];
        for(unsigned int i=0; i < anglecorrinv[d].size(); ++i)
            anglecorrinvfile << setw(13) << anglecorrinv[d][i];
        for(unsigned int i=0; i < distcorrhor[d].size(); ++i)
            distcorrhorfile << setw(13) << distcorrhor[d][i];
        for(unsigned int i=0; i < distcorrver[d].size(); ++i)
            distcorrverfile << setw(13) << distcorrver[d][i];
        for(unsigned int i=0; i < distcorrinv[d].size(); ++i)
            distcorrinvfile << setw(13) << distcorrinv[d][i];

        topnhorfile << endl;
        topnverfile << endl;
        topninvfile << endl;
        meanangleshorfile << endl;
        meananglesverfile << endl;
        meananglesinvfile << endl;
        meandisthorfile << endl;
        meandistverfile << endl;
        meandistinvfile << endl;
        anglecorrhorfile << endl;
        anglecorrverfile << endl;
        anglecorrinvfile << endl;
        distcorrhorfile << endl;
        distcorrverfile << endl;
        distcorrinvfile << endl;
    }

    topnhorfile.close();
    topnverfile.close();
    topninvfile.close();
    meanangleshorfile.close();
    meananglesverfile.close();
    meananglesinvfile.close();
    meandisthorfile.close();
    meandistverfile.close();
    meandistinvfile.close();
    anglecorrhorfile.close();
    anglecorrverfile.close();
    anglecorrinvfile.close();
    distcorrhorfile.close();
    distcorrverfile.close();
    distcorrinvfile.close();
    ///////////////////////////////////////////

    SaveLabel2LabelRelations(&f, outputdir + "/label2label");


    std::vector<std::vector<double> > bla(maxdepth+2, std::vector<double>(f.GetNrOfLabels(), 0.0));

    for(int tidx=0; tidx < f.GetNrOfTrees(); ++tidx)
    {
        v4r::EntangledForestTree* t = f.GetTree(tidx);

        for(int nidx=0; nidx < t->GetNrOfNodes(); ++nidx)
        {
            v4r::EntangledForestNode* n = t->GetNode(nidx);
            int d = n->GetDepth();
            std::transform(bla[d].begin(), bla[d].end(), n->GetLabelDistribution().begin(), bla[d].begin(), std::plus<double>());
        }
    }

    ofstream reldistfile(outputdir + "/reldistfile");
    for(unsigned int d=0; d<bla.size(); ++d)
    {
        for(unsigned int i=0; i<bla[d].size(); ++i)
        {
            reldistfile << setw(13) << bla[d][i];
        }
        reldistfile << std::endl;
    }
    reldistfile.close();
}
