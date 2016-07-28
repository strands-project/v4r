/******************************************************************************
 * Copyright (c) 2015 Martin Velas
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

/**
*
*      @author martin Velas (ivelas@fit.vutbr.cz)
*      @date Jan, 2016
*      @brief multiview object instance recognizer with change detection filtering
*/

#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/flann_search.hpp>

#include <v4r/recognition/multiview_object_recognizer_change_detection.h>
#include <v4r/change_detection/miscellaneous.h>
#include <v4r/change_detection/change_detection.h>

namespace po = boost::program_options;

namespace v4r
{

template<typename PointT>
MultiviewRecognizerWithChangeDetection<PointT>::MultiviewRecognizerWithChangeDetection(int argc, char ** argv) :
        MultiviewRecognizer<PointT>(argc, argv), changing_scene(new Cloud) {
    po::options_description desc("");
    desc.add_options()
            ("min_points_for_hyp_removal", po::value<int>(&param_.min_points_for_hyp_removal_)->default_value(param_.min_points_for_hyp_removal_), "Minimal overlap with removed points in order to remove hypothesis")
            ("tolerance_for_cloud_diff", po::value<float>(&param_.tolerance_for_cloud_diff_)->default_value(param_.tolerance_for_cloud_diff_), "Tolerance when difference of clouds is computed")
    ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; exit(0); }
    try { po::notify(vm); }
    catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
}

template<typename PointT>
void
MultiviewRecognizerWithChangeDetection<PointT>::initHVFilters () {
    View<PointT> &v = views_[id_];
    removed_points_history_[id_].reset(new Cloud);
    removed_points_cumulated_history_[id_].reset(new Cloud);

    Cloud added_points_ignore;
    findChangedPoints(*v.scene_, Eigen::Affine3f(v.absolute_pose_), *(removed_points_history_[id_]), added_points_ignore);

    // fill-in the cumulated history of changes:
    typename std::map < size_t, CloudPtr >::iterator rem_it;
    for (rem_it = removed_points_cumulated_history_.begin(); rem_it != removed_points_cumulated_history_.end(); rem_it++) {
        *rem_it->second += *removed_points_history_[id_];
    }

    /* // store visualization of the results
    changes_visualization.clear();
    pcl::transformPointCloud(*v.scene_, changes_visualization, v.absolute_pose_);
    v4r::VisualResultsStorage::copyCloudColored(*removed_points_history_[id_], changes_visualization, 255, 0, 0);
    v4r::VisualResultsStorage::copyCloudColored(*added_points_, changes_visualization, 0, 255, 0);
    visResStore.savePcd("changes", changes_visualization);
    changes_visualization.clear();
    v4r::VisualResultsStorage::copyCloudColored(*removed_points_history_[id_], changes_visualization, 255, 0, 0);
    v4r::VisualResultsStorage::copyCloudColored(*added_points_, changes_visualization, 0, 255, 0);
    */
}

template<typename PointT>
void
MultiviewRecognizerWithChangeDetection<PointT>::findChangedPoints(
        Cloud observation_unposed,
        Eigen::Affine3f pose,
        Cloud &removed_points,
        Cloud &added_points) {

    CloudPtr observation(new Cloud);
    pcl::transformPointCloud(observation_unposed, *observation, pose);
    observation = v4r::downsampleCloud<PointT>(observation);
    if(!changing_scene->empty()) {
        v4r::ChangeDetector<PointT> detector;
        detector.detect(changing_scene, observation, pose, param_.tolerance_for_cloud_diff_);
      v4r::ChangeDetector<PointT>::removePointsFrom(changing_scene, detector.getRemoved());
      removed_points += *(detector.getRemoved());
        added_points += *(detector.getAdded());
        *changing_scene += added_points;
    } else {
        added_points += *observation;
        *changing_scene += *observation;
    }
}

template<typename PointT>
void
MultiviewRecognizerWithChangeDetection<PointT>::reconstructionFiltering(typename pcl::PointCloud<PointT>::Ptr observation,
        pcl::PointCloud<pcl::Normal>::Ptr observation_normals, const Eigen::Matrix4f &absolute_pose, size_t view_id) {
    CloudPtr observation_transformed(new Cloud);
    pcl::transformPointCloud(*observation, *observation_transformed, absolute_pose);
    CloudPtr cloud_tmp(new Cloud);
    std::vector<int> indices;
    v4r::ChangeDetector<PointT>::difference(observation_transformed, removed_points_cumulated_history_[view_id],
            cloud_tmp, indices, param_.tolerance_for_cloud_diff_);

    /* Visualization of changes removal for reconstruction:
    Cloud rec_changes;
    rec_changes += *cloud_transformed;
    v4r::VisualResultsStorage::copyCloudColored(*removed_points_cumulated_history_[view_id], rec_changes, 255, 0, 0);
    v4r::VisualResultsStorage::copyCloudColored(*cloud_tmp, rec_changes, 200, 0, 200);
    stringstream ss;
    ss << view_id;
    visResStore.savePcd("reconstruction_changes_" + ss.str(), rec_changes);*/

    std::vector<bool> preserved_mask(observation->size(), false);
    for (std::vector<int>::iterator i = indices.begin(); i < indices.end(); i++) {
        preserved_mask[*i] = true;
    }
    for (size_t j = 0; j < preserved_mask.size(); j++) {
        if (!preserved_mask[j]) {
            setNan(observation->at(j));
            setNan(observation_normals->at(j));
        }
    }
    PCL_INFO("Points by change detection removed: %d\n", observation->size() - indices.size());
}

template<typename PointT>
void
MultiviewRecognizerWithChangeDetection<PointT>::setNan(pcl::Normal &normal) {
   const float nan_point = std::numeric_limits<float>::quiet_NaN();
   normal.normal_x = nan_point;
   normal.normal_y = nan_point;
   normal.normal_z = nan_point;
}

template<typename PointT>
void
MultiviewRecognizerWithChangeDetection<PointT>::setNan(PointT &pt) {
   const float nan_point = std::numeric_limits<float>::quiet_NaN();
   pt.x = nan_point;
   pt.y = nan_point;
   pt.z = nan_point;
}

template<typename PointT>
std::vector<bool>
MultiviewRecognizerWithChangeDetection<PointT>::getHypothesisInViewsMask(ModelTPtr model, const Eigen::Matrix4f &pose, size_t origin_id) {

    std::vector<size_t> sorted_view_ids = getMapKeys(views_);
    std::sort(sorted_view_ids.begin(), sorted_view_ids.end());

    size_t hypothesis_removed_since_view = id_ + 1; // assuming never removed (id_+1 = future)
    for(std::vector<size_t>::iterator view_id = sorted_view_ids.begin(); view_id < sorted_view_ids.end(); view_id++) {
        if(*view_id <= origin_id) {
            continue; // hypothesis can not be removed before discovered
        }
        if(removed_points_history_[*view_id]->empty()) {
            continue; // no changes were found in this view
        }

        CloudPtr model_aligned(new Cloud);
        pcl::transformPointCloud(*model->assembled_, *model_aligned, pose);
        int rem_support = v4r::ChangeDetector<PointT>::removalSupport(
            removed_points_history_[*view_id], model_aligned, param_.tolerance_for_cloud_diff_, 1.1)->size();

        PCL_INFO("Support for the removal of object [%s,%d]: %d\n", model->id_.c_str(), *view_id, rem_support);
        if(rem_support > param_.min_points_for_hyp_removal_) {
            hypothesis_removed_since_view = *view_id;
        }
    }

    std::vector<bool> mask;
    for(auto view : views_) {
        mask.push_back(view.first >= origin_id && view.first < hypothesis_removed_since_view);
    }
    return mask;
}

template<typename PointT>
template<typename K, typename V>
std::vector<K> MultiviewRecognizerWithChangeDetection<PointT>::getMapKeys(const std::map<K, V> &container) {
    std::vector<K> keys;
    for(auto it: container) {
        keys.push_back(it.first);
    }
    return keys;
}

template<typename PointT>
void
MultiviewRecognizerWithChangeDetection<PointT>::cleanupHVFilters() {
    std::vector<size_t> view_ids_vector = getMapKeys(views_);
    std::set<size_t> view_ids_set(view_ids_vector.begin(), view_ids_vector.end());

    std::vector<size_t> removed_ids = getMapKeys(removed_points_history_);
    for(size_t id_to_remove : removed_ids) {
        if(view_ids_set.find(id_to_remove) == view_ids_set.end()) {
            removed_points_history_.erase(id_to_remove);
            removed_points_cumulated_history_.erase(id_to_remove);
        }
    }
}

}
