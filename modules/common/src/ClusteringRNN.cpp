/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#include <v4r/common/ClusteringRNN.h>

namespace v4r
{

using namespace std;


ClusteringRNN::ClusteringRNN(const Parameter &_param, bool _dbg)
 : param(_param), dbg(_dbg)
{
}

ClusteringRNN::~ClusteringRNN()
{
}




/************************************** PRIVATE ************************************/





/************************************** PUBLIC ************************************/


/**
 * find nearest neighbour of CodebookEntries
 */
int ClusteringRNN::getNearestNeighbour(const Cluster &cluster, const std::vector<Cluster::Ptr> &clusters, float &sim)
{
  sim = -FLT_MAX;
  int idx = INT_MAX;
  float tmp;

  for (unsigned i=0; i<clusters.size(); i++)
  {
    tmp = -( cluster.sqr_sigma + clusters[i]->sqr_sigma + 
             (cluster.data-clusters[i]->data).squaredNorm() );
    if (tmp > sim)
    {
      sim = tmp;
      idx = i;
    }
  }

  return idx;
}

/**
 * Agglomerate
 */
void ClusteringRNN::agglomerate(const Cluster &src, Cluster &dst)
{
  float sum = 1. / ( src.indices.size() + dst.indices.size() );

  dst.sqr_sigma = sum * (
              src.indices.size()*src.sqr_sigma +
              dst.indices.size()*dst.sqr_sigma +
              sum*src.indices.size()*dst.indices.size()*(src.data-dst.data).squaredNorm() );

  //compute new mean model of two clusters
  dst.data *= dst.indices.size();

  dst.data += src.data*src.indices.size();
  dst.data *= sum;
  
  //add occurrences from c
  dst.indices.insert( dst.indices.end(), src.indices.begin(), src.indices.end() );
}

/**
 * initDataStructure
 */
void ClusteringRNN::initDataStructure(const DataMatrix2Df &samples, std::vector< Cluster::Ptr > &data)
{
  data.resize(samples.rows);

  for (int i=0; i<samples.rows; i++)
  {
    data[i].reset( new Cluster(Eigen::Map<const Eigen::VectorXf>(&samples(i,0), samples.cols),i) );
  }
}


/**
 * create clusters
 */
void ClusteringRNN::cluster(const DataMatrix2Df &samples)
{
  int nn, last;
  float sim;
  std::vector<float> lastsim;
  std::vector< Cluster::Ptr > chain;
  std::vector< Cluster::Ptr > remaining;

  initDataStructure(samples, remaining);  

  clusters.clear();

  if (remaining.size()==0)
    return;

  last=0;
  lastsim.push_back(-FLT_MAX);

  chain.push_back(remaining.back());
  remaining.pop_back();
  float sqrThr = -param.dist_thr*param.dist_thr;

  while (remaining.size()!=0){
    nn = getNearestNeighbour(*chain[last], remaining, sim);

    if(sim > lastsim[last]){
      //no RNN -> add to chain
      last++;
      chain.push_back(remaining[nn]);
      remaining.erase(remaining.begin()+nn);
      lastsim.push_back(sim);
    } else {
      //RNN found
      if (lastsim[last] > sqrThr){
        agglomerate(*chain[last-1], *chain[last]);
        remaining.push_back(chain[last]);
        chain.pop_back();
        chain.pop_back();
        lastsim.pop_back();
        lastsim.pop_back();
        last-=2;
      }else{
        //cluster found set codebook
        for (unsigned i=0; i<chain.size(); i++){
          clusters.push_back(chain[i]);
        }
        chain.clear();
        lastsim.clear();
        last=-1;
        if (dbg){ printf("."); fflush(stdout); }
      }
    }

    if (last<0){
      //init new chain
      last++;
      lastsim.push_back(-FLT_MAX);

      chain.push_back(remaining.back());
      remaining.pop_back();
    }
  }

  for (unsigned i=0; i<chain.size(); i++){
    clusters.push_back(chain[i]);
  }  

  if (dbg) cout<<endl;
  //if (dbg) cout<<"clusters.size()="<<clusters.size()<<endl;
}

/**
 * getClusters
 */
void ClusteringRNN::getClusters(std::vector<std::vector<int> > &_clusters)
{
  _clusters.resize(clusters.size());

  for (unsigned i=0; i<clusters.size(); i++)
    _clusters[i] = clusters[i]->indices;
}

/**
 * getCenters
 */
void ClusteringRNN::getCenters(DataMatrix2Df &_centers)
{
  _centers.clear();

  if (clusters.size()==0)
    return;

  int cols = clusters[0]->data.size();
  _centers.reserve(clusters.size(), cols);

  for (unsigned i=0; i<clusters.size(); i++)
    _centers.push_back(&clusters[i]->data[0], cols);
}


}







