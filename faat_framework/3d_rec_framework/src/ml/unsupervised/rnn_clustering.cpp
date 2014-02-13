/*
 * rnn_clustering.cpp
 *
 *  Created on: Apr 23, 2012
 *      Author: aitor
 */

#include <faat_pcl/3d_rec_framework/ml/unsupervised/rnn_clustering.h>
#include <fstream>

void
faat_pcl::rec_3d_framework::RNNClustering::cluster ()
{
  int nn, last;
  float sim;
  std::vector<float> lastsim;
  std::vector < CBEntry > chain;
  std::vector < CBEntry > remaining;

  clusters_.clear();

  remaining.resize(features_.size());

  //create CBEntries
  for(size_t i=0; i < features_.size(); i++) {
    remaining[i].mean_ = features_[i];
    remaining[i].sqrSigma_ = 0.f;
    remaining[i].occurrences_ = 1;
  }

  last = 0;
  lastsim.push_back (-std::numeric_limits<float>::max ());

  chain.push_back (remaining.back ());
  remaining.pop_back ();
  float sqrThr = -(t_ * t_);

  while (remaining.size () != 0)
  {
    nn = GetNearestNeighbour (chain[last], remaining, sim);

    if (sim > lastsim[last])
    {
      //no RNN -> add to chain
      last++;
      chain.push_back (remaining[nn]);
      remaining.erase (remaining.begin () + nn);
      lastsim.push_back (sim);
    }
    else
    {
      //RNN found
      if (lastsim[last] > sqrThr)
      {
        Agglomerate (chain[last - 1], chain[last]);
        remaining.push_back (chain[last]);
        chain.pop_back ();
        chain.pop_back ();
        lastsim.pop_back ();
        lastsim.pop_back ();
        last -= 2;
      }
      else
      {
        //cluster found set codebook
        for (unsigned i = 0; i < chain.size (); i++)
        {
          clusters_.push_back (chain[i]);
        }
        chain.clear ();
        lastsim.clear ();
        last = -1;

        std::cout << "." << std::flush;
      }
    }

    if (last < 0)
    {
      //init new chain
      last++;
      lastsim.push_back (-std::numeric_limits<float>::max ());

      chain.push_back (remaining.back ());
      remaining.pop_back ();
    }
  }

  std::cout << std::endl << clusters_.size() << std::endl;
}

void faat_pcl::rec_3d_framework::RNNClustering::load_cb_disk(std::string & path, std::vector<Eigen::VectorXf> & cb) {
  std::ifstream in;
  in.open (path.c_str (), std::ifstream::in);

  char linebuf[4096];
  while(in.getline (linebuf, 4096)) {
    std::string line (linebuf);
    std::vector < std::string > strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));
    Eigen::VectorXf cluster;
    cluster.setZero(strs_2.size() - 1);
    for(size_t i=0; i < (strs_2.size() - 1); i++) {
      cluster[i] = atof(strs_2[i].c_str());
    }

    cb.push_back(cluster);
  }
}

void faat_pcl::rec_3d_framework::RNNClustering::save_cb_disk(std::string & path) {
  std::ofstream out (path.c_str ());
  if (!out)
  {
    std::cout << "Cannot open file.\n";
  }

  for(size_t i=0; i < clusters_.size(); i++) {
    for(size_t j=0; j < clusters_[i].mean_.size(); j++) {
      out << clusters_[i].mean_[j];
      if(j < clusters_[i].mean_.size()) {
        out << " ";
      }
    }

    out << std::endl;
  }

  out.close ();
}

