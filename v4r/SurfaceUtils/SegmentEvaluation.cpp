/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
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
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */

/**
 * @file SegmentEvaluation.cpp
 * @author Richtsfeld
 * @date Dezember 2012
 * @version 0.1
 * @brief Evaluation of the segmentation results.
 */

#include "SegmentEvaluation.h"

namespace surface
{


/************************************************************************************
 * Constructor/Destructor
 */

SegmentEvaluation::SegmentEvaluation()
{
  gpAnno = false;
  consider_nan_values = false;
}

SegmentEvaluation::~SegmentEvaluation()
{}

// ================================= Private functions ================================= //


// ================================= Public functions ================================= //

void SegmentEvaluation::compute()
{
  static double sum_tp = 0;
  static double sum_fp = 0;
  static double sum_fn = 0;
  
  double img_tp = 0;
  double img_fp = 0;
  double img_fn = 0;

  /// create object image
  std::vector<int> objects;                   
  objects.resize(pcl_cloud_labeled->points.size());
  for(unsigned i=0; i<objects.size(); i++)
    objects[i] = -1;
  for(unsigned i=0; i < view->surfaces.size(); i++)
    for(unsigned j=0; j< view->surfaces[i]->indices.size(); j++)
      objects[view->surfaces[i]->indices[j]] = view->surfaces[i]->label;

  /// get number of ground truth objects
  unsigned nr_gt_objects = 0;
  for(unsigned i=0; i<pcl_cloud_labeled->points.size(); i++)
    if(pcl_cloud_labeled->points[i].label > nr_gt_objects)
      nr_gt_objects = pcl_cloud_labeled->points[i].label;
  nr_gt_objects++;
  unsigned gt[nr_gt_objects];
  for(unsigned i=0; i<nr_gt_objects; i++)
    gt[i] = 0;
  for(unsigned i=0; i<pcl_cloud_labeled->points.size(); i++)
    if(consider_nan_values || !isnan(pcl_cloud_labeled->points[i].x))
      gt[pcl_cloud_labeled->points[i].label]++;

// for(unsigned i=0; i<nr_gt_objects; i++)
// printf("gt[%u]: %u\n", i, gt[i]);

  /// get number of objects in surfaces
  unsigned nr_objects = 0;
  for(unsigned i=0; i<view->surfaces.size(); i++)
    if(view->surfaces[i]->label > nr_objects)
      nr_objects = view->surfaces[i]->label;
  nr_objects++;
  unsigned object[nr_objects];
  for(unsigned i=0; i< nr_objects; i++)
    object[i]=0;
  for(unsigned i=0; i<view->surfaces.size(); i++)
    object[view->surfaces[i]->label] += view->surfaces[i]->indices.size();
  
// for(unsigned i=0; i<nr_objects; i++)
// printf("object[%u]: %u\n", i, object[i]);
  
// printf("Number of gt objects: %u - number of objects: %u\n", nr_gt_objects, nr_objects);

  /// Open files for writing
  static int filenumber = 0;
  FILE *seg_results = std::fopen("SegmentationResults.txt", "a");
  FILE *obj_results = std::fopen("SegmentationResultsObject.txt", "a");
  std::fprintf(seg_results,"File with number: %u  (tp, fn, fp, Recall, Fos=1-Recall, Fus, Precision) \n", filenumber);
  filenumber++;
    
  /// find best match
  unsigned obj;
  if(gpAnno)
    obj = 2;
  else
    obj = 1;

  for(obj; obj<nr_gt_objects; obj++) {       /// TODO obj=0 ==> Background? / obj=1 ==> ground-plane


    if(gt[obj] == 0) // ground truth model is empty => skip
      continue;

    unsigned object_support[nr_objects];
    for(unsigned i=0; i<nr_objects; i++)
      object_support[i]=0;
    
    for(unsigned idx=0; idx< objects.size(); idx++)
      if(pcl_cloud_labeled->points[idx].label == obj)
        object_support[objects[idx]]++;
    
    /// best object support
    int tp_i = 0;
    int best = 0;
    for(unsigned i=0; i<nr_objects; i++) {
      if(object_support[i] > tp_i) {
        best = i;
        tp_i = object_support[i];
      }
    }
   
    int fn_i = gt[obj] - tp_i;
    int fp_i = object[best] - tp_i;

    double tp = (double) tp_i;
    double fn = (double) fn_i;
    double fp = (double) fp_i;

    img_tp += tp;
    img_fn += fn;
    img_fp += fp;
    std::fprintf(seg_results,"  object %u: %6.0f, %6.0f, %6.0f,  %6.3f, %6.3f, %6.3f, %6.3f\n", obj, tp, fn, fp, 100.*tp/(tp+fn), 100.*fn/(tp+fn), 100.*fp/(tp+fn), 100.*tp/(tp+fp));
    std::fprintf(obj_results,"%6.3f %6.3f\n", tp/(tp+fp), tp/(tp+fn));
  }
  std::fprintf(seg_results,"  image:  %6.0f, %6.0f, %6.0f, %6.3f, %6.3f, %6.3f, %6.3f\n", img_tp, img_fn, img_fp, 100.*img_tp/(img_tp+img_fn), 100.*img_fn/(img_tp+img_fn), 100.*img_fp/(img_tp+img_fn), 100.*img_tp/(img_tp+img_fp));

  /// overall results
  sum_tp += img_tp;
  sum_fn += img_fn;
  sum_fp += img_fp;
  std::fprintf(seg_results,"  overall: %6.0f, %6.0f, %6.0f, %6.3f, %6.3f, %6.3f, %6.3f\n\n", sum_tp, sum_fn, sum_fp, 100.*sum_tp/(sum_tp+sum_fn), 100.*sum_fn/(sum_tp+sum_fn), 100.*sum_fp/(sum_tp+sum_fn), 100.*sum_tp/(sum_tp+sum_fp));
  std::fclose(seg_results);
  std::fclose(obj_results);
  
  
  
  /// TODO HACK AddOn: Check, if Graph-Cut results are better than the binary results from the SVM => 
  static int changed = 0, counter = 0;
  static int tp=0, tn=0, fp=0, fn=0;
  // get each relation
  for(unsigned i=0; i<view->relations.size(); i++) {
// printf("Relation: %u-%u: gt: %u => %4.3f\n", view->relations[i].id_0, view->relations[i].id_1, view->relations[i].groundTruth, view->relations[i].rel_probability[1]);
    bool svm, gc;
    // check probability value >, < 0.5 => Check against ground truth
    if(view->relations[i].rel_probability[1] > 0.5)
      svm = true;
    else
      svm = false;
    
    // compare labels from results => Check against ground truth
    if(view->surfaces[view->relations[i].id_0]->label == view->surfaces[view->relations[i].id_1]->label)
      gc = true;
    else
      gc = false;
    

    if(view->relations[i].groundTruth != -1 && view->relations[i].groundTruth == svm)                   /// Woher kommt die Ground-Truth??? => beim Testen immer aus ersten, beim Lernen aus BEIDEN
      if(view->relations[i].groundTruth == 1)
        tp++;
      else
        tn++;
    if(view->relations[i].groundTruth != -1 && view->relations[i].groundTruth != svm)
      if(view->relations[i].groundTruth == 1)
        fn++;
      else
        fp++;
      
    if(view->relations[i].groundTruth != -1)
      counter++;
    if(view->relations[i].groundTruth != -1 && view->relations[i].groundTruth != svm && svm != gc)      // svm war falsch und graphCut sagt aber was anderes
      changed++;
  }
    
  double tpf = (double) tp;
  double tnf = (double) tn;
  double fpf = (double) fp;
  double fnf = (double) fn;
  printf("tp, tn, fp, fn: %u, %u, %u, %u\n", tp, tn, fp, fn);
  printf("BER: %4.3f / changed: %u\n", 0.5*((fpf/(fpf+tpf))+(fnf/(fnf+tnf))), changed);
}


}


