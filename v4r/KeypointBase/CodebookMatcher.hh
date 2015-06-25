/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_CODEBOOK_MATCHER_HH
#define KP_CODEBOOK_MATCHER_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointTools/ClusteringRNN.hh"
#include "v4r/KeypointTools/triple.hpp"

namespace kp 
{

class CodebookMatcher
{
public:
  class Parameter
  {
  public:
    float thr_desc_rnn;
    float nnr;
    float max_dist;
    Parameter(float _thr_desc_rnn=0.55, float _nnr=0.92, float _max_dist=.7)
    : thr_desc_rnn(_thr_desc_rnn), nnr(_nnr), max_dist(_max_dist) {}
  };

private:
  Parameter param;

  ClusteringRNN rnn;

  int max_view_index;
  DataMatrix2Df descs;
  std::vector< std::pair<int,int> > vk_indices;

  cv::Mat cb_centers;
  std::vector< std::vector< std::pair<int,int> > > cb_entries;
  std::vector< std::pair<int, int> > view_rank;

  cv::Ptr<cv::DescriptorMatcher> matcher;

public:
  cv::Mat dbg;

  CodebookMatcher(const Parameter &p=Parameter());
  ~CodebookMatcher();

  void clear();
  void addView(const cv::Mat &descriptors, int view_idx);
  void createCodebook();
  void createCodebook(cv::Mat &_cb_centers, std::vector< std::vector< std::pair<int,int> > > &_cb_entries);
  void setCodebook(const cv::Mat &_cb_centers, const std::vector< std::vector< std::pair<int,int> > > &_cb_entries);
  void queryViewRank(const cv::Mat &descriptors, std::vector< std::pair<int, int> > &view_rank);
  void queryMatches(const cv::Mat &descriptors, std::vector< std::vector< cv::DMatch > > &matches);

  inline const std::vector< std::vector< std::pair<int,int> > > &getEntries() { return cb_entries; }
  inline const cv::Mat &getDescriptors() { return cb_centers; }
  inline const std::vector< std::pair<int, int> > &getViewRank() {return view_rank;}

  typedef SmartPtr< ::kp::CodebookMatcher> Ptr;
  typedef SmartPtr< ::kp::CodebookMatcher const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

