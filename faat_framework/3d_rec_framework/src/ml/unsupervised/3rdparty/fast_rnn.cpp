/*
 * fast_rnn.cpp
 *
 *  Created on: Feb 14, 2013
 *      Author: aitor
 */

//fast-RNN v2.0
//    Copyright (C) 2012  Roberto J. López-Sastre (robertoj.lopez@uah.es)
//                        Daniel Oñoro-Rubio
//                        Víctor Carrasco-Valdelvira
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "faat_pcl/3d_rec_framework/ml/unsupervised/3rdparty/fast_rnn.h"

namespace fast_rnn
{

  /*void centroid_mul (double n, vector<double> & centroid, vector<double> & out);
  void centroid_plus (vector<double> & A, vector<double> & B, vector<double> & centroid);
  void centroid_div (double n, vector<double> & centroid, vector<double> & out);
  void centroid_diff (vector<double> & A, vector<double> & B, vector<double> & out);*/

  //vector<double>
  void fastRNN::centroid_mul (double n, vector<double> & centroid, vector<double> & out)
  {
    int size = centroid.size ();
    out.resize(size);
    for (int i = 0; i < size; i++)
      out[i] = (centroid[i] * n);

    //return centroid;
  }

  void
  fastRNN::centroid_plus (vector<double> & A, vector<double> & B, vector<double> & centroid)
  {
    int size = A.size ();
    centroid.resize(size);
    for (int i = 0; i < size; i++)
      centroid[i] = (A[i] + B[i]);

  }

  //vector<double>
  void
  fastRNN::centroid_div (double n, vector<double> & centroid, vector<double> & out)
  {
    int size = centroid.size ();
    out.resize(size);
    for (int i = 0; i < size; i++)
      out[i] = (centroid[i] / n);
      //centroid[i] = centroid[i] / n;

    //return centroid;
  }

  //vector<double>
  void
  fastRNN::centroid_diff (vector<double> & A, vector<double> & B, vector<double> & centroid)
  {
    int size = A.size ();
    centroid.resize(size);
    for (int i = 0; i < size; i++)
      centroid[i] = (A[i] - B[i]);

    //return centroid;
  }

  double
  fastRNN::squared_magnitude (vector<double> & vec)
  {
    int size = vec.size ();
    double sum = 0;
    for (int i = 0; i < size; i++)
      sum += vec[i] * vec[i];
    return sum;
  }

  void
  fastRNN::agglomerate_clusters (cluster &C1, cluster &C2)
  {
    //Agglomerate two clusters
    //C1=C1+C2

    unsigned int m = C2.data_index.size ();
    unsigned int n = C1.data_index.size ();
    double d_sum = double (n + m);

    //Copy index values
    C1.data_index.reserve(m);
    for (unsigned int i = 0; i < m; i++)
      C1.data_index.push_back (C2.data_index[i]);

    //update centroid
    vector<double> centroid_ps, centroid_mul1, centroid_mul2;
    centroid_mul (double (n), C1.centroid, centroid_mul1);
    centroid_mul (double (m), C2.centroid, centroid_mul2);
    centroid_plus (centroid_mul1, centroid_mul2, centroid_ps);
    centroid_div (d_sum, centroid_ps, C1.centroid);

    //update variance
    vector<double> diff;
    centroid_diff (C1.centroid, C2.centroid, diff);
    C1.cvar = ((double (n) * C1.cvar) + (double (m) * C2.cvar) + (((double (n * m)) / (d_sum)) * (squared_magnitude (diff)))) / (d_sum);

  }

  //Get Nearest Neighbor
  void
  fastRNN::get_nn (cluster &C, list<cluster> &R, double &sim, list<cluster>::iterator &iNN)
  {
    //Return the NN of cluster C in the list R.
    //iNN is the iterator of the NN in R, and sim is the similarity.

    unsigned int n = R.size ();
    list<cluster>::iterator it, itBegin = R.begin (), itEnd = R.end ();
    double d;
    vector<double> diff; //, diff2;

    if (n > 0)
    {
      //First iteration
      it = itBegin;
      //diff = centroid_diff (C.centroid, (*it).centroid);
      centroid_diff (C.centroid, (*it).centroid, diff);
      sim = -(C.cvar + (*it).cvar + squared_magnitude (diff));
      iNN = it;
      it++;

      for (; it != itEnd; it++)
      {
        //diff = centroid_diff (C.centroid, (*it).centroid);
        centroid_diff (C.centroid, (*it).centroid, diff);

        d = -(C.cvar + (*it).cvar + squared_magnitude (diff));

        if (d > sim)
        {
          sim = d;
          iNN = it;
        }
      }
    }
    else
    {
      cout << "Warning: R is empty (function: get_nn)" << endl;
      sim = 0;
    }
  }

  void
  fastRNN::get_nn_in_slices (cluster &C, list<cluster> &X, slice &Si, slice &So, double &sim, list<cluster>::iterator &iNN, double limit)
  {
    //Search within the interior and exterior slices, Si and So respectively
    double d;
    slice::iterator itS, endS;
    list<cluster>::iterator it;
    vector<double> diff;
    bool isfirst = true;

    //Search first in the interior slice
    if (Si.size () > 0)
    {
      endS = Si.end ();
      //First iteration
      isfirst = false;
      itS = Si.begin ();
      it = (*itS).it;
      //diff = centroid_diff (C.centroid, (*it).centroid);
      centroid_diff (C.centroid, (*it).centroid, diff);
      sim = -(C.cvar + (*it).cvar + squared_magnitude (diff));
      iNN = it;

      itS++;
      for (; itS != endS; itS++)
      {
        it = (*itS).it;
        //diff = centroid_diff (C.centroid, (*it).centroid);
        centroid_diff (C.centroid, (*it).centroid, diff);
        d = -(C.cvar + (*it).cvar + squared_magnitude (diff));
        if (d > sim)
        {
          sim = d;
          iNN = it;
        }
      }

      //DEBUG
      ndist += Si.size ();

      //Do we need to search in the exterior slice?
      if (sim >= limit)
      {
        return; //NO
      }
    }

    //Search in the exterior slice (if any)
    if (So.size () > 0)
    {
      endS = So.end ();
      if (isfirst)
      {
        //First iteration
        isfirst = false;
        itS = So.begin ();
        it = (*itS).it;
        //diff = centroid_diff (C.centroid, (*it).centroid);
        centroid_diff (C.centroid, (*it).centroid, diff);
        sim = -(C.cvar + (*it).cvar + squared_magnitude (diff));
        iNN = it;
      }

      for (itS = So.begin (); itS != endS; itS++)
      {
        it = (*itS).it;
        //diff = centroid_diff (C.centroid, (*it).centroid);
        centroid_diff (C.centroid, (*it).centroid, diff);
        d = -(C.cvar + (*it).cvar + squared_magnitude (diff));
        if (d > sim)
        {
          sim = d;
          iNN = it;
        }
      }

      //DEBUG
      ndist += So.size ();
    }
  }

  int unsigned
  fastRNN::bsearchL (fmap &f_map, double d)
  {
    //Search in f_map to the left
    int b = 0, c, t = f_map.size () - 1, aux_c;
    double q;
    bool end = false;

    // Move until we find a top value not erased
    while (!f_map[t].mask)
      t--;

    // Move until we find a base value not erased
    while (!f_map[b].mask)
      b++;

    while (((t - b) > 1) && !end)
    {
      //the middle
      c = (b + t) >> 1;

      //erased
      if (!f_map[c].mask)
      {
        // Search a not erased element

        // Fitst iteration
        aux_c = c + 1;

        //Searching upward
        while (!f_map[aux_c].mask && (aux_c < t))
          aux_c++;

        //Do we need to search downward?
        if (!f_map[aux_c].mask || (aux_c >= t))
        {
          aux_c = c - 1;

          // Searching downward
          while (!f_map[aux_c].mask && (aux_c > b))
            aux_c--;

        }

        if (aux_c == b)
          end = true;
        else
          c = aux_c;

      }//if erased

      if (!end && !f_map[c].mask)
      {
        cout << "bsearchL failed" << endl;
        exit (-1);
      }

      if (!end)
      {
        q = (*f_map[c].it).centroid[COMP];

        if (d < q)
          t = c;
        else if (d > q)
          b = c;
        else
          return c;

      }
    }

    if (!f_map[b].mask || !f_map[t].mask)
    {
      cout << "Error: bsearchL failed" << endl;
      exit (-1);
    }

    return ((d <= (*f_map[b].it).centroid[COMP]) ? b : t);

  }

  unsigned int
  fastRNN::bsearchR (fmap &f_map, double d)
  {
    int b = 0, c, t = f_map.size () - 1, aux_c;
    double q;
    bool end = false;

    // Move until find a top not erased
    while (!f_map[t].mask)
      t--;

    // Move until find a base not erased
    while (!f_map[b].mask)
      b++;

    while ((t - b > 1) && !end)
    {

      c = (b + t) >> 1;

      //erased
      if (!f_map[c].mask)
      {
        aux_c = c + 1;

        // Search upward
        while (!f_map[aux_c].mask && (aux_c < t))
          aux_c++;

        //Do we need to search downward
        if (!f_map[aux_c].mask || (aux_c >= t))
        {
          aux_c = c - 1;

          // Search downward
          while (!f_map[aux_c].mask && (aux_c > b))
            aux_c--;
        }

        if (aux_c == b)
          end = true;
        else
          c = aux_c;
      }

      if (!end && !f_map[c].mask)
      {
        cout << "error: bsearchR failed" << endl;
        exit (-1);
      }

      if (!end)
      {
        q = (*f_map[c].it).centroid[COMP];
        if (d < q)
          t = c;
        else if (d > q)
          b = c;
        else
          return c;
      }
    }

    if (!f_map[b].mask || !f_map[t].mask)
    {
      cout << "error: bsearchR failed" << endl;
      exit (-1);
    }

    return ((d >= (*f_map[t].it).centroid[COMP]) ? t : b);
  }

  int
  fastRNN::bsearch (fmap &f_map, double d, unsigned int &b, unsigned int &t)
  {
    unsigned int leng = f_map.size () - 1;
    b = 0;
    t = leng;
    unsigned int c, aux_c;
    double q;
    bool end = false;

    //highest no deleted position
    while ((t > 0) && !f_map[t].mask)
      t--;
    //lowest no deleted position
    while ((b < leng) && !f_map[b].mask)
      b++;

    //Check conditions
    if (b > t)
      return -1;

    //Binary search
    while (((t - b) > 1) && !end)
    {
      c = (b + t) >> 1;

      //is it erased?
      if (!f_map[c].mask)
      {
        aux_c = c + 1;

        // Search upward
        while (!f_map[aux_c].mask && (aux_c < t))
          aux_c++;

        // Do we have to search downward?
        if (!f_map[aux_c].mask || (aux_c >= t))
        {
          aux_c = c - 1;

          // Search downward
          while (!f_map[aux_c].mask && (aux_c > b))
            aux_c--;
        }

        if (aux_c == b)
          end = true;
        else
          c = aux_c;
      }

      if (!end && !f_map[c].mask)
      {
        cout << "error: bsearch error" << endl;
        exit (-1);
      }

      if (!end)
      {
        q = (*f_map[c].it).centroid[COMP];
        if (d < q)
          t = c;
        else if (d > q)
          b = c;
        else
          return c;
      }
    }

    if (!f_map[b].mask || !f_map[t].mask)
    {
      cout << "error: bsearch error" << endl;
      exit (-1);
    }

    if (b == t)
      return ((*f_map[b].it).centroid[COMP] >= d) ? t : t + 1;
    else
      return ((*f_map[b].it).centroid[COMP] >= d) ? b : t;

  }

  void
  fastRNN::init_slices (fmap &f_map, list<cluster> &X, slice &Si, slice &So, double V, double e)
  {
    //Generate the slice in the space where the NN candidates are. The slice has 2e width.
    unsigned int min, max, bmax, bmin, tmax, tmin, i;
    candidate c;

    //Three slices? (recall: ep is the parameter for slicing)
    if (e > ep)
    {
      //Build interior slice
      min = bsearchL (f_map, V - ep);
      max = bsearchR (f_map, V + ep);

      for (i = min; i <= max; i++)
      {
        if (f_map[i].mask)
        {
          c.it = f_map[i].it;
          c.index = i;
          Si.push_back (c);
        }
      }

      if (min != 0) //generate bottom candidate list
      {
        bmax = min - 1;
        //Build bottom slice
        bmin = bsearchL (f_map, V - e);

        for (i = bmin; i <= bmax; i++)
        {
          if (f_map[i].mask)
          {
            c.it = f_map[i].it;
            c.index = i;
            So.push_back (c);
          }
        }
      }

      if (max != (f_map.size () - 1))
      {
        tmin = max + 1;

        //Build top slice
        tmax = bsearchR (f_map, V + e);

        for (i = tmin; i <= tmax; i++)
        {
          if (f_map[i].mask)
          {
            c.it = f_map[i].it;
            c.index = i;
            So.push_back (c);
          }
        }
      }
    }
    else //only one slice
    {
      min = bsearchL (f_map, V - e);
      max = bsearchR (f_map, V + e);

      for (i = min; i <= max; i++)
      {
        if (f_map[i].mask)
        {
          c.it = f_map[i].it;
          c.index = i;
          Si.push_back (c);
        }
      }
    }
  }

  void
  fastRNN::erase_element (fmap &f_map, list<cluster> &X, list<cluster>::iterator it)
  {
    int l = f_map.size (), i;

    for (i = 0; i < l; i++)
    {
      if (f_map[i].mask)
        if (f_map[i].it == it)
        {
          f_map[i].mask = false;
          X.erase (it);

          //Update free_top
          free_top = (i > free_top) ? i : free_top;
          return;
        }
    }

    cout << "error: erasing element" << endl;
    exit (-1);
  }

  void
  fastRNN::insert_element (fmap &f_map, list<cluster> &X, cluster &V)
  {
    //Insert element V in X and update f_map
    element elem2insert, aux_elem;
    int pos = 0;
    unsigned int b, t;
    bool update_free_top;

    //Push back the element in X
    X.push_back (V);

    // Initialize the element to insert
    elem2insert.mask = true;

    elem2insert.it = X.end ();
    elem2insert.it--;

    //Search for a position
    pos = bsearch (f_map, V.centroid[COMP], b, t);

    // f_map is empty
    if (-1 == pos)
    {
      pos = t;
      f_map[pos] = elem2insert;
      return;
    }

    //Is pos the last position?
    if (f_map.size () == (unsigned int)pos)
    {
      pos--;
      if (!f_map[pos].mask)
      {
        f_map[pos] = elem2insert;
        return;
      }

      //Insert downwards

      while (elem2insert.mask)
      {

        //Save current pos element
        aux_elem = f_map[pos];

        //Insert element in pos
        f_map[pos] = elem2insert;

        //Update elem2insert
        elem2insert = aux_elem;

        pos--;

      }
      free_top = pos;
      return;

    }

    if (pos >= free_top)
    {
      //Insert downwards
      while (elem2insert.mask)
      {
        pos--;
        //Save current pos element
        aux_elem = f_map[pos];

        //Insert element in pos
        f_map[pos] = elem2insert;

        //Update elem2insert
        elem2insert = aux_elem;
      }
      //update free_top?
      update_free_top = true;

    }
    else //upwards
    {
      if (f_map[pos].mask)
        if (V.centroid[COMP] >= (*f_map[pos].it).centroid[COMP])
          pos++;

      while (elem2insert.mask)
      {
        //Save current pos element
        aux_elem = f_map[pos];

        //Insert element in pos
        f_map[pos] = elem2insert;

        //Update elem2insert
        elem2insert = aux_elem;

        //Update index
        pos++;
      }

      //update free_top?
      update_free_top = (pos < free_top) ? false : true;

    }

    //Update free_top just in case it has been occupied
    if (update_free_top)
    {
      free_top = pos - 1;
      while ((free_top > 0) && (f_map[free_top].mask))
        free_top--;
    }

  }

  void
  fastRNN::init_map (fmap &f_map, list<cluster> &X)
  {

    //Create forward map
    list<cluster>::iterator itX, endX;
    list<Point<list<cluster>::iterator> > aux_list;
    list<Point<list<cluster>::iterator> >::iterator it, itend;
    element eaux;

    //Get the list of iterators
    endX = X.end ();
    for (itX = X.begin (); itX != endX; itX++)
      aux_list.push_back (Point<list<cluster>::iterator> (itX, (*itX).centroid[COMP]));

    //Sorting
    aux_list.sort ();

    eaux.mask = true; //Mask = true for all elements

    //Convert the sorted list to f_map
    itend = aux_list.end ();
    for (it = aux_list.begin (); it != itend; it++)
    {
      eaux.it = (*it).get_index (); //save the iterators
      f_map.push_back (eaux);
    }

    //Update free top
    free_top = -1;
  }

  unsigned int
  fastRNN::agg_clustering_fast_rnn (matrix_data &X, vector<unsigned int> &labels, double agg_thres, vector<vector<double> > &cluster_centre)
  {
    //This function computes the Fast RNN (reciprocal nearest neighbors) clustering.

    //Chain for the clusters R
    list<cluster> R (n);
    list<cluster>::iterator it, itEnd = R.end (), iNN, penult;
    matrix_data::iterator Xit = X.begin ();
    unsigned int Xindex = 0;
    double sim = 0;
    double l_agg_thres = (-1 * agg_thres);
    double epsilon;
    slice Si, So; //slices with candidates
    bool RNNfound = false;

    //DEBUG
    ndist = 0;

    //Initialize list R - each vector in a separate cluster (start point of the algorithm)
    for (it = R.begin (); it != itEnd; it++, Xit++, Xindex++)
    {
      //update index of the vector
      (*it).data_index.push_back (Xindex);
      //update centroid
      (*it).centroid = *Xit;
      //update variance
      (*it).cvar = 0;
    }

    //NN-Chain (the pair at the end of this chain is always a RNN)
    list<cluster> L;
    //Chain for similarities
    list<double> Lsim;
    //chain for the (final) clusters C
    list<cluster> C;

    //Create forward map
    fmap f_map;
    init_map (f_map, R);

    //The algorithm starts with a random cluster
    srand(
    time (NULL));
    unsigned int rp = (unsigned int)rand () % n; //random integer [0,n-1]


    //Add to L
    it = R.begin ();
    advance (it, rp);
    L.push_back (*it);

    //R\rp -> delete the cluster in R and mark as erased in f_map
    erase_element (f_map, R, it);

    //First iteration
    if (R.size () > 0)
    {
      //Get nearest neighbor
      get_nn (L.back (), R, sim, iNN);

      //DEBUG
      ndist += R.size ();

      //Add to the NN chain
      L.push_back (*iNN); //add to L
      erase_element (f_map, R, iNN); //delete from R
      Lsim.push_back (sim);//add to Lsim


      //Only two clusters?
      if (R.size () == 0)
      {
        penult = L.end ();
        penult--;
        penult--;
        //check the similarity (last element)
        if (sim > l_agg_thres)
        {
          //Agglomerate clusters
          agglomerate_clusters (L.back (), *penult);
          C.push_back (L.back ());
        }
        else
        {
          //Save in C separately
          C.push_back (*penult);
          C.push_back (L.back ());
        }
        L.clear (); //free memory
      }
    }
    else //R is empty
    {
      if (L.size () == 1)
        C.push_back (L.back ()); // Only one vector, only one cluster

      L.clear (); //free memory
    }
    //Main loop
    while (R.size () > 0)
    {
      RNNfound = false;

      //Clear slices
      Si.clear ();
      So.clear ();

      //Update epsilon with the last sim
      epsilon = sqrt (-1 * Lsim.back ());

      epsilon = (epsilon < ep) ? epsilon : ep;

      //Identify slices
      init_slices (f_map, R, Si, So, L.back ().centroid[COMP], epsilon);

      if ((Si.size () > 0) || (So.size () > 0)) //Search for a NN within the candidate list
      {

        get_nn_in_slices (L.back (), R, Si, So, sim, iNN, l_agg_thres);

        if (sim > Lsim.back ()) //no RNN
        {

          //No RNNs, add s to the NN chain
          L.push_back (*iNN); //add to L
          erase_element (f_map, R, iNN); //delete from R
          Lsim.push_back (sim);//add to Lsim


          if (R.size () == 0) //R has been emptied
          {
            //check the last similarity
            if (Lsim.back () > l_agg_thres)
            {
              //Agglomerate clusters
              penult = L.end ();
              penult--;
              penult--;
              agglomerate_clusters (L.back (), *penult);
              insert_element (f_map, R, L.back ());

              //delete the last two elements in L
              L.pop_back ();
              L.pop_back ();

              //delete similarities
              Lsim.pop_back ();
              if (Lsim.size () >= 1)
                Lsim.pop_back ();

              //Initialize the chain with the following nearest neighbour
              if (L.size () == 1)
              {
                //Get nearest neighbor
                get_nn (L.back (), R, sim, iNN);

                ndist += R.size ();

                //Add to the NN chain
                L.push_back (*iNN); //add to L
                erase_element (f_map, R, iNN); //delete from R
                Lsim.push_back (sim);//add to Lsim


                if (R.size () == 0) //R has been emptied?
                {
                  penult = L.end ();
                  penult--;
                  penult--;

                  //check the similarity
                  if (Lsim.back () > l_agg_thres)
                  {
                    //Agglomerate clusters
                    agglomerate_clusters (L.back (), *penult);
                    C.push_back (L.back ()); //add the cluster to C
                  }
                  else
                  {
                    //Save in C
                    C.push_back (*penult);
                    C.push_back (L.back ());
                  }
                  break;//end main while
                }
              }
            }
            else
            {
              //Add the clusters to C (separate clusters)
              itEnd = L.end ();
              for (it = L.begin (); it != itEnd; it++)
                C.push_back (*it);
              break;//end main while
            }
          }
        }
        else
          //A RNN
          RNNfound = true;
      }

      //RNN found
      if (RNNfound || ((Si.size () == 0) && (So.size () == 0)))
      {

        if (Lsim.back () > l_agg_thres) //can they be agglomerated?
        {
          //Agglomerate clusters
          penult = L.end ();
          penult--;
          penult--;
          agglomerate_clusters (L.back (), *penult);

          insert_element (f_map, R, L.back ());

          L.pop_back ();
          L.pop_back ();

          //delete similarities
          Lsim.pop_back ();
          if (Lsim.size () >= 1)
            Lsim.pop_back ();

          if (L.size () == 1)
          {
            //Get nearest neighbor
            get_nn (L.back (), R, sim, iNN);

            //DEBUG
            ndist += R.size ();

            //Add the NN chain
            //Add to the NN chain
            L.push_back (*iNN); //add to L
            erase_element (f_map, R, iNN); //delete from R
            Lsim.push_back (sim);//add to Lsim


            if (R.size () == 0) //R has been emptied?
            {
              penult = L.end ();
              penult--;
              penult--;
              //check the similarity
              if (Lsim.back () > l_agg_thres)
              {
                //Agglomerate clusters
                agglomerate_clusters (L.back (), *penult);
                C.push_back (L.back ()); //add the cluster to C

              }
              else
              {
                //Save in C
                C.push_back (*penult);
                C.push_back (L.back ());
              }

              break;
            }
          }
        }
        else //discard this chain
        {
          //Add the clusters to C (separate clusters)
          itEnd = L.end ();
          for (it = L.begin (); it != itEnd; it++)
            C.push_back (*it);

          L.clear ();
        }
      }

      //Do we need to start a new chain?
      if (L.size () == 0)
      {
        //Initialize a new chain
        Lsim.clear ();

        //random point
        srand(
        time (NULL));
        rp = rand () % R.size (); //random point


        //Add to L
        it = R.begin ();
        advance (it, rp);
        L.push_back (*it);

        //R\rp -> delete the cluster in R and mark as erased in f_map
        erase_element (f_map, R, it);

        //First iteration
        if (R.size () > 0)
        {
          //Get nearest neighbor
          get_nn (L.front (), R, sim, iNN);

          //DEBUG
          ndist += R.size ();

          //Add to the NN chain
          L.push_back (*iNN); //add to L
          erase_element (f_map, R, iNN); //delete from R
          Lsim.push_back (sim);//add to Lsim


          //Only two clusters?
          if (R.size () == 0)
          {
            penult = L.end ();
            penult--;
            penult--;

            //check the similarity (last element)
            if (Lsim.back () > l_agg_thres)
            {
              //Agglomerate clusters
              agglomerate_clusters (L.back (), *penult);
              C.push_back (L.back ());
            }
            else
            {
              //Save in C separately
              C.push_back (*penult);
              C.push_back (L.back ());
            }
            break;//end main while
          }
        }
        else //R is empty
        {
          if (L.size () == 1)
            C.push_back (L.front ());
        }
      }
    }

    //Chain C contains all the clusters
    nc = C.size ();//number of clusters

    if (nc > 0)
      dim = (C.front ()).centroid.size (); // take the dimension from the fitst element in C

    cluster_centre.clear ();//delete the content

    vector<double> aux_centroid (dim);

    itEnd = C.end ();
    unsigned int c_label = 0, num_labels = 0, s;

    for (it = C.begin (); it != itEnd; it++)
    {
      //convert from vcl to vnl
      for (s = 0; s < dim; s++)
        aux_centroid[s] = (*it).centroid[s];

      //add centroid
      cluster_centre.push_back (aux_centroid);

      num_labels += (*it).data_index.size ();

      for (s = 0; s < (*it).data_index.size (); s++)
        labels[(*it).data_index[s]] = c_label;

      c_label++;
    }

    //Were all the points asigned?.
    if (num_labels != n)
    {
      cout << "Warning: all the points were not assigned to a cluster!" << endl;
      cout << "Num. Labels = " << num_labels << " Num. Points = " << n << endl;
    }

    return nc;
  }

  fastRNN::fastRNN() {
    thres_euclid = 0.125;
    thres = thres_euclid * thres_euclid;
    ep = 0.005;
    COMP = 0;
  }

  void
  fastRNN::setData (std::vector<std::vector<double> > & data)
  {
    n = data.size ();
    dim = data[0].size();
    for(size_t i=0; i < n; i++) {
      X.push_back(data[i]);
    }
  }

  void
  fastRNN::do_clustering ()
  {
    labels_data.resize (n);
    nc = agg_clustering_fast_rnn (X, labels_data, thres, cluster_centre);
    std::cout << "Number of clusters:" << nc << " cluster centers size:" << cluster_centre.size() << std::endl;
  }

  void fastRNN::getCentersAndAssignments(vector<vector<float> > & cluster_centers, vector<unsigned int> & assignments)
  {
    cluster_centers.resize(cluster_centre.size());
    for(size_t i=0; i < cluster_centre.size(); i++) {
      cluster_centers[i] = std::vector<float>( cluster_centre[i].begin(), cluster_centre[i].end() );
    }

    assignments = labels_data;
  }

  int fastRNN::save_data(std::string & out_file_path)
  {
    unsigned int i=0,k=0;

    //Out file
    ofstream ffile(out_file_path.c_str());

    if( ffile.is_open() )
    {
      //Write num_distances
      ffile << dim; ffile << endl;
      ffile << n; ffile << endl;

      //Write labels first
      for(k = 0; k < n; k++)
      {
        ffile << labels_data[k];
        ffile << endl;
      }

      //Write number of clusters
      ffile << nc << endl;

      //Write centers
      for( i=0; i < nc; i++)
      {
        for(k=0; k<dim; k++)
          ffile << cluster_centre[i][k] << " ";

        ffile << endl;

      }
    }
    else
    {
      cout << "Error writing results file." << endl;
      return -1;
    }

    //close file
    ffile.close();
    return 1;
  }

  int fastRNN::load_data(std::string & out_file_path)
  {
    unsigned int i=0,k=0;

    //Out file
    ifstream ffile(out_file_path.c_str());

    if( ffile.is_open() )
    {
      //Write num_distances
      ffile >> dim;
      ffile >> n;

      //Write labels first
      labels_data.resize(n);
      for(k = 0; k < n; k++)
      {
        ffile >> labels_data[k];
      }

      //Write number of clusters
      ffile >> nc;

      //Write centers
      cluster_centre.resize(nc);
      for( i=0; i < nc; i++)
      {
        cluster_centre[i].resize(dim);
        for(k=0; k<dim; k++)
          ffile >> cluster_centre[i][k];
      }
    }
    else
    {
      cout << "Error reading codebook." << endl;
      return -1;
    }

    //close file
    ffile.close();
    return 1;
  }

} //namespace

////////////////////////////////
//           MAIN
///////////////////////////////

/*int main(int argc, char *argv[])
 {
 //DEBUG TIME
 float t_rnn=0;


 list< vector<double> > mylist;


 //Read options from command line
 get_options(argc,argv);

 cout << endl;
 cout << "================" << endl;
 cout << "    fast-RNN    " << endl;
 cout << "================" << endl << endl;

 //list with all the vectors
 matrix_data X;

 //vector for centres
 vector< vector<double> > cluster_centre;

 //vector for labels
 vector <unsigned int> labels_data(n);

 cout << "Reading data ...";

 //read data from file
 read_data(X);

 cout << "[Done]" << endl;

 cout << "fast-RNN ...";

 //fast-rnn clustering
 TIME_THIS(
 nc=agg_clustering_fast_rnn(X, labels_data, thres, cluster_centre),
 t_rnn);

 cout << "[Done in " << t_rnn << " sec]" << endl;

 cout << "Saving data ...";

 save_data(labels_data,cluster_centre,t_rnn,ndist);

 cout << "[Done]" << endl;

 //Free memory, we will not need it anymore!
 X.clear();
 labels_data.clear();
 cluster_centre.clear();

 cout << "ThE eNd" << endl;

 return 0;
 }*/
