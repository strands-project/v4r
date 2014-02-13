/*
 * rnn_clustering.h
 *
 *  Created on: Apr 23, 2012
 *      Author: aitor
 */

#ifndef RNN_CLUSTERING_H_
#define RNN_CLUSTERING_H_

#include <pcl/common/common.h>
#include <boost/algorithm/string.hpp>

namespace faat_pcl
{
  namespace rec_3d_framework
  {
    class RNNClustering
    {
    private:

      class CBEntry
      {
      public:
        int occurrences_;
        Eigen::VectorXf mean_;
        float sqrSigma_;
      };

      float t_;
      std::vector<Eigen::VectorXf> features_;
      std::vector<CBEntry> clusters_;
    public:
      RNNClustering ()
      {
        t_ = 0.1f;
      }

      void
      setThreshold (float t)
      {
        t_ = t;
      }

      void
      setFeatures (std::vector<Eigen::VectorXf> & f)
      {
        features_ = f;
      }

      void
      getCodebook (std::vector<Eigen::VectorXf> & cb)
      {
        for (size_t i = 0; i < clusters_.size (); i++)
          cb.push_back (clusters_[i].mean_);
      }

      int
      GetNearestNeighbour (CBEntry & cbe, std::vector<CBEntry> & cb, float & sim)
      {
        int idx = std::numeric_limits<int>::max ();
        float sim2;
        sim = -std::numeric_limits<float>::max ();

        for (unsigned i = 0; i < cb.size (); i++)
        {
          //sim2 = -(cbe * cbe - cb[i] * cb[i]).sum ();
          Eigen::VectorXf diff = cbe.mean_ - cb[i].mean_;
          sim2 = -(cbe.sqrSigma_ + cb[i].sqrSigma_ + (diff.cwiseProduct (diff)).sum ());
          if (sim2 > sim)
          {
            sim = sim2;
            idx = i;
          }
        }

        return idx;
      }

      void
      Agglomerate (CBEntry & src, CBEntry & dst)
      {
        float sum = static_cast<float> (src.occurrences_ + dst.occurrences_);
        Eigen::VectorXf diff = src.mean_ - dst.mean_;
        dst.sqrSigma_ = 1. / sum * (static_cast<float> (src.occurrences_) * src.sqrSigma_ + static_cast<float> (dst.occurrences_) * dst.sqrSigma_
            + static_cast<float> (src.occurrences_ * dst.occurrences_) / sum * static_cast<float> (diff.cwiseProduct (diff).sum ()));

        //compute new mean model of two clusters
        dst.mean_ *= static_cast<float> (dst.occurrences_);
        src.mean_ *= static_cast<float> (src.occurrences_);
        dst.mean_ += src.mean_;
        dst.mean_ *= 1.f / sum;

        dst.occurrences_ += src.occurrences_;
      }

      void
      cluster ();

      void
      save_cb_disk (std::string & path);

      void
      load_cb_disk (std::string & path, std::vector<Eigen::VectorXf> & cb);

    };
  }
}

#endif /* RNN_CLUSTERING_H_ */
