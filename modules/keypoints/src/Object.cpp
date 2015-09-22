
#include <v4r/keypoints/impl/Object.hpp>


namespace v4r
{



/******************** impl ObjectView **************************/
/** get global 3d points **/
void ObjectView::getPoints(std::vector<Eigen::Vector3f> &pts)
{
  pts.resize(points.size());
  for (unsigned i=0; i<pts.size(); i++)
    pts[i] = getPt(i).pt.cast<float>();
}

 
/** get global normals **/
void ObjectView::getNormals(std::vector<Eigen::Vector3f> &normals)
{
  normals.resize(points.size());
  for (unsigned i=0; i<normals.size(); i++)
    normals[i] = getPt(i).n.cast<float>();
}


/* add a keypoint */
void ObjectView::add(const cv::KeyPoint &key, float *d, int dsize, const Eigen::Vector3f &pt3, const Eigen::Vector3f &n, const Eigen::Vector3f &vr, const Eigen::Vector3f &cam_pt, const int &part_idx) {
  keys.push_back(key);
  if (descs.empty()) cv::Mat(1,dsize,CV_32F,d).copyTo(descs);
  else descs.push_back(cv::Mat(1,dsize,CV_32F,d));
  if (object==0) std::runtime_error("[ObjectView::add] No object membership available!");
  else points.push_back(object->addPt(pt3,n));
  viewrays.push_back(vr);
  cam_points.push_back(cam_pt);
  part_indices.push_back(part_idx);
}

/* add a keypoint */
void ObjectView::add(const cv::KeyPoint &key, float *d, int dsize, unsigned glob_idx, const Eigen::Vector3f &vr, const Eigen::Vector3f &cam_pt, const int &part_idx) {
  keys.push_back(key);
  if (descs.empty()) cv::Mat(1,dsize,CV_32F,d).copyTo(descs);
  else descs.push_back(cv::Mat(1,dsize,CV_32F,d));
  if (object==0) std::runtime_error("[ObjectView::add] No object membership available!");
  else points.push_back(object->incPt(glob_idx)); 
  viewrays.push_back(vr);
  cam_points.push_back(cam_pt);
  part_indices.push_back(part_idx);
}

/** delete a complete entry **/
void ObjectView::del(unsigned idx)
{
  if (idx<keys.size()) keys.erase(keys.begin()+idx);
  if (idx<points.size()) {
    object->decPt(points[idx]);  
    points.erase(points.begin()+idx);
  }
  if (idx<viewrays.size()) viewrays.erase(viewrays.begin()+idx);
  if (idx<cam_points.size()) cam_points.erase(cam_points.begin()+idx);
  if (idx<projs.size()) projs.erase(projs.begin()+idx);
  if (idx<part_indices.size()) part_indices.erase(part_indices.begin()+idx);

  std::cout<<"[ObjectView::delPt] TODO: delete the descriptor!"<<std::endl;
}
 

/* compute center */
void ObjectView::computeCenter() {
  if (points.size()>0 && object!=0) {
    center = Eigen::Vector3f(0.,0.,0.);
    for (unsigned i=0; i<points.size(); i++)
      center += getPt(i).pt.cast<float>();
    center /= float(points.size());
  } else center = Eigen::Vector3f::Zero();
}

/* copy to (projs are not copied!) */
void ObjectView::copyTo(ObjectView &view) {
  view.idx = idx;
  view.camera_id = camera_id;
  view.center = center;
  image.copyTo(view.image);
  descs.copyTo(view.descs);
  view.keys = keys;
  view.points = points;
  view.viewrays = viewrays;
  view.cam_points = cam_points;
  view.part_indices = part_indices;
  view.object = object;
}

/** clear **/
void ObjectView::clear() {
  camera_id = -1;
  image = cv::Mat_<unsigned char>();
  descs = cv::Mat();
  keys.clear();
  points.clear();
  viewrays.clear();
  cam_points.clear();
  projs.clear();
  part_indices.clear();
}



} //--END--

