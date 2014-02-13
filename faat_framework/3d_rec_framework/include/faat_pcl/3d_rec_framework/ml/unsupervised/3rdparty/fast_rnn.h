/*
 * fast_rnn.h
 *
 *  Created on: Feb 14, 2013
 *      Author: aitor
 */

#ifndef FAST_RNN_H_
#define FAST_RNN_H_

#include <fstream>
#include <vector>
#include <list>
#include <sys/time.h>
#include <stdlib.h>
#include <limits>
#include <string.h>
#include <math.h>
#include "Point.h"

using namespace std;

namespace fast_rnn
{
  class fastRNN
  {
  private:

    typedef struct scluster
    {
      vector<double> centroid; //centroid
      vector<unsigned int> data_index; //index of vectors of this cluster
      double cvar; //cluster varianceabout:startpage
    } cluster;

    typedef list<vector<double> > matrix_data;

    //element structure
    typedef struct selement
    {
      list<cluster>::iterator it;
      bool mask;
    } element;

    typedef vector<element> fmap;

    //Struct of candidates to nn
    typedef struct scandidate
    {
      list<cluster>::iterator it;
      unsigned int index;
    } candidate;

    typedef list<candidate> slice;

    void centroid_mul (double n, vector<double> & centroid, vector<double> & out);
    void centroid_plus (vector<double> & A, vector<double> & B, vector<double> & centroid);
    void centroid_div (double n, vector<double> & centroid, vector<double> & out);
    void centroid_diff (vector<double> & A, vector<double> & B, vector<double> & out);
    double
    squared_magnitude (vector<double> &);
    void
    agglomerate_clusters (cluster &C1, cluster &C2);
    void
    get_nn (cluster &C, list<cluster> &R, double &sim, list<cluster>::iterator &iNN);
    void
    get_nn_in_slices (cluster &C, list<cluster> &X, slice &Si, slice &So, double &sim, list<cluster>::iterator &iNN, double limit);
    int unsigned
    bsearchL (fmap &f_map, double d);
    unsigned int
    bsearchR (fmap &f_map, double d);
    int
    bsearch (fmap &f_map, double d, unsigned int &b, unsigned int &t);
    void
    init_slices (fmap &f_map, list<cluster> &X, slice &Si, slice &So, double V, double e);
    void
    erase_element (fmap &f_map, list<cluster> &X, list<cluster>::iterator it);
    void
    insert_element (fmap &f_map, list<cluster> &X, cluster &V);
    void
    init_map (fmap &f_map, list<cluster> &X);
    unsigned int
    agg_clustering_fast_rnn (matrix_data &X, vector<unsigned int> &labels, double agg_thres, vector<vector<double> > &cluster_centre);

    unsigned int dim;
    unsigned int n;
    double thres; //threshold for agglomerative clustering
    double thres_euclid;
    double ep; //threshold for slicing
    unsigned int nc; //num clusters
    int free_top;
    unsigned int COMP;
    unsigned long ndist;
    matrix_data X;
    vector<vector<double> > cluster_centre;
    vector<unsigned int> labels_data;
  public:
    fastRNN ();
    void
    setData (std::vector<std::vector<double> > & data);
    void
    do_clustering ();

    void getCentersAndAssignments(vector<vector<float> > & cluster_centers, vector<unsigned int> & assignments);
    int load_data(std::string & out_file_path);
    int save_data(std::string & out_file_path);



  };
}

#endif /* FAST_RNN_H_ */
