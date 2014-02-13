// #include "Symmetry3DMap.hpp"
// 
// namespace AttentionModule
// {
// 
// Symmetry3DMapParameters::Symmetry3DMapParameters()
// {
//   cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
//   normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
//   indices = pcl::PointIndices::Ptr(new pcl::PointIndices());
//   normalization_type = EPUtils::NT_NONE;
//   R1 = 5;
//   R2 = 11;
//   S = 16;
//   filter_size = 5;
//   map = cv::Mat_<float>::zeros(0,0);
//   width = 0;
//   height = 0;
//   cameraParametrs.clear();
// }
// 
// void Calculate3DSymmetryPCA(AttentionModule::Symmetry3DMapParameters &parameters)
// {
//   cv::Mat depth;
//   EPUtils::convertPointCloud2Depth(parameters.cloud,depth,parameters.width,parameters.height);
//   
//   int R2 = parameters.R2;
//   R2 = 30;
//   
//   // create normals map
//   cv::Mat lookupTableNormals = cv::Mat_<int>::zeros(parameters.height,parameters.width);
//   lookupTableNormals = lookupTableNormals - 1;
//   for(unsigned int pi = 0; pi < parameters.cloud->size(); ++pi)
//   {
//     pcl::PointXYZ p = parameters.cloud->points.at(pi);
//     
//     //int r = (int)(p.normal_x);
//     //int c = (int)(p.normal_y);
//     int r = pi / parameters.width;
//     int c = pi % parameters.width;
//     
//     lookupTableNormals.at<int>(r,c) = pi;
//     
//   }
//   
//   std::vector<cv::Point> points_ul;
//   std::vector<cv::Point> points_ur;
//   std::vector<cv::Point> points_dl;
//   std::vector<cv::Point> points_dr;
//   
//   // create pairs of points
//   for(int rr = 0; rr < R2/2; ++rr)
//   {
//     for(int cc = 0; cc < R2/2; ++cc)
//     {
//       int distr = R2/2 - rr;
//       int distc = R2/2 - cc;
//       
//       cv::Point temp;
//       
//       temp.y = -distr;
//       temp.x = -distc;
//       points_ul.push_back(temp);
//       
//       temp.y = distr;
//       temp.x = distc;
//       points_dr.push_back(temp);
//       
//       temp.y = -distr;
//       temp.x = distc;
//       points_ur.push_back(temp);
//       
//       temp.y = distr;
//       temp.x = -distc;
//       points_dl.push_back(temp);
//     }
//   }
//   
//   parameters.map = cv::Mat_<float>::zeros(parameters.height,parameters.width);
//   
//   for(int rr = R2/2; rr < parameters.height - R2/2; ++rr)
//   {
//     for(int cc = R2/2; cc < parameters.width - R2/2; ++cc)
//     {
//       
//       if(depth.at<float>(rr,cc) > 0)
//       {
//         pcl::PointCloud<pcl::Normal> normals;
// 	pcl::PointCloud<pcl::PointXYZ> small_cloud;
// 	
// 	int index = lookupTableNormals.at<int>(rr,cc);
// 	
// 	pcl::Normal current_normal = parameters.normals->points.at(index);
// 	
// 	int points_ul_num = 0;
// 	int points_ur_num = 0;
// 	int points_dl_num = 0;
// 	int points_dr_num = 0;
// 	
// 	for(unsigned int pi = 0; pi < points_ul.size(); ++pi)
//         {
// 	
// 	  int r1 = rr + points_ul.at(pi).y;
// 	  int c1 = cc + points_ul.at(pi).x;
// 	  
// 	  int index1 = lookupTableNormals.at<int>(r1,c1);
// 	  
// 	  if(index1 >= 0)
// 	  {
// 	    small_cloud.points.push_back(parameters.cloud->points.at(index1));
// 	    normals.points.push_back(parameters.normals->points.at(index1));
// 	    points_ul_num += 1;
// 	  }
// 	}
// 	
// 	for(unsigned int pi = 0; pi < points_ur.size(); ++pi)
//         {
// 	
// 	  int r1 = rr + points_ur.at(pi).y;
// 	  int c1 = cc + points_ur.at(pi).x;
// 	  
// 	  int index1 = lookupTableNormals.at<int>(r1,c1);
// 	  
// 	  if(index1 >= 0)
// 	  {
// 	    small_cloud.points.push_back(parameters.cloud->points.at(index1));
// 	    normals.points.push_back(parameters.normals->points.at(index1));
// 	    points_ur_num += 1;
// 	  }
// 	}
// 	
// 	for(unsigned int pi = 0; pi < points_dl.size(); ++pi)
//         {
// 	
// 	  int r1 = rr + points_dl.at(pi).y;
// 	  int c1 = cc + points_dl.at(pi).x;
// 	  
// 	  int index1 = lookupTableNormals.at<int>(r1,c1);
// 	  
// 	  if(index1 >= 0)
// 	  {
// 	    small_cloud.points.push_back(parameters.cloud->points.at(index1));
// 	    normals.points.push_back(parameters.normals->points.at(index1));
// 	    points_dl_num += 1;
// 	  }
// 	}
// 	
// 	for(unsigned int pi = 0; pi < points_dr.size(); ++pi)
//         {
// 	
// 	  int r1 = rr + points_dr.at(pi).y;
// 	  int c1 = cc + points_dr.at(pi).x;
// 	  
// 	  int index1 = lookupTableNormals.at<int>(r1,c1);
// 	  
// 	  if(index1 >= 0)
// 	  {
// 	    small_cloud.points.push_back(parameters.cloud->points.at(index1));
// 	    normals.points.push_back(parameters.normals->points.at(index1));
// 	    points_dr_num += 1;
// 	  }
// 	}
// 	
// 	small_cloud.width = small_cloud.points.size();
// 	small_cloud.height = 1;
// 	
// 	if((small_cloud.points.size() <= 0) || (normals.points.size() <= 0))
// 	  continue;
// 	
// 	//if((points_ul_num < points_ul.size()/2) || (points_ur_num < points_ur.size()/2) || (points_dl_num < points_dl.size()/2) || (points_dr_num < points_dr.size()/2))
// 	  //continue;
// 	  
// 	std::vector<pcl::Normal> axis;
// 	//std::cerr << "normals.size() = " << normals.size() << std::endl;
// 	EPUtils::principleAxis(normals,axis);
// 	
// 	std::vector<float> W;
// 	W.resize(axis.size());
// 	
// 	for(unsigned int axis_num = 0; axis_num < axis.size(); ++axis_num)
// 	{
// 	  W.at(axis_num) = 0;
// 	  
// 	  // create plane
// 	  pcl::Normal plane_normal;
// 	  plane_normal = axis.at(axis_num);
// 	  EPUtils::normalizeVector(plane_normal);
// 	
// 	  float x0 = parameters.cloud->points.at(index).x;
// 	  float y0 = parameters.cloud->points.at(index).y;
// 	  float z0 = parameters.cloud->points.at(index).z;
// 	
// 	  float a = plane_normal.normal[0];
// 	  float b = plane_normal.normal[1];
// 	  float c = plane_normal.normal[2];
// 	  float d = -(a*x0 + b*y0 + c*z0);
// 	
// 	  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
// 	  coefficients->values.resize(4);
// 	  coefficients->values.at(0) = a;
// 	  coefficients->values.at(1) = b;
// 	  coefficients->values.at(2) = c;
// 	  coefficients->values.at(3) = d;
// 	  
// 	  // display point cloud and normals
// 	/*pcl::PointCloud<pcl::PointXYZ>::Ptr small_cloud_XYZ;
// 	EPUtils::convertPointXYZRGBRC2PointXYZ(small_cloud.makeShared(),small_cloud_XYZ);
// 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//         viewer->setBackgroundColor (0, 0, 0);
//         viewer->addPointCloud<pcl::PointXYZ> (small_cloud_XYZ, "small_cloud_XYZ");
//         viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "small_cloud_XYZ");
//         viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (small_cloud_XYZ, normals.makeShared(), 10, 0.02, "normals");
// 	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "normals");
// 	
// 	viewer->addPlane(coefficients);
// 	while (!viewer->wasStopped ())
//         {
//           viewer->spinOnce (100);
//           boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//         }*/
// 
//           pcl::PointCloud<pcl::PointXYZ>::Ptr points_projected (new pcl::PointCloud<pcl::PointXYZ>);
// 	  std::vector<float> distances;
// 	  EPUtils::ProjectPointsOnThePlane(coefficients,small_cloud.makeShared(),points_projected,distances,false);
// 	
// 	  int left_num = 0;
// 	  float left_distance = 0;
// 	  pcl::Normal left_normal;
// 	  left_normal.normal[0] = 0;
// 	  left_normal.normal[1] = 0;
// 	  left_normal.normal[2] = 0;
// 	  pcl::PointXYZ left_point;
// 	  left_point.x = 0;
// 	  left_point.y = 0;
// 	  left_point.z = 0;
// 	  int right_num = 0;
// 	  float right_distance = 0;
// 	  pcl::Normal right_normal;
// 	  right_normal.normal[0] = 0;
// 	  right_normal.normal[1] = 0;
// 	  right_normal.normal[2] = 0;
// 	  pcl::PointXYZ right_point;
// 	  right_point.x = 0;
// 	  right_point.y = 0;
// 	  right_point.z = 0;
// 	
// 	/*pcl::PointCloud<pcl::PointXYZ>::Ptr points_left (new pcl::PointCloud<pcl::PointXYZ>());
// 	pcl::PointCloud<pcl::PointXYZ>::Ptr points_right (new pcl::PointCloud<pcl::PointXYZ>());
// 	pcl::PointCloud<pcl::Normal>::Ptr normals_left (new pcl::PointCloud<pcl::Normal>());
// 	pcl::PointCloud<pcl::Normal>::Ptr normals_right (new pcl::PointCloud<pcl::Normal>());*/
// 	    
// 	  
// 	  for(unsigned int pi = 0; pi < small_cloud.size(); ++pi)
//  	  {
// 	    float point_x = small_cloud.points.at(pi).x;
// 	    float point_y = small_cloud.points.at(pi).y;
// 	    float point_z = small_cloud.points.at(pi).z;
// 	    
// 	    float dist_to_the_plane = 0;
// 	    
// 	    pcl::Normal p1p2;
// 	    p1p2.normal[0] = point_x - x0;
// 	    p1p2.normal[1] = point_y - y0;
// 	    p1p2.normal[2] = point_z - z0;
// 	    
// 	    dist_to_the_plane = p1p2.normal[0]*plane_normal.normal[0] + p1p2.normal[1]*plane_normal.normal[1] + p1p2.normal[2]*plane_normal.normal[2];
// 	
// 	    if(dist_to_the_plane > 0)
// 	    {
// 	      left_num += 1;
// 	      left_distance += distances.at(pi);
// 	    
// 	      left_normal.normal[0] += normals.points.at(pi).normal[0];
// 	      left_normal.normal[1] += normals.points.at(pi).normal[1];
// 	      left_normal.normal[2] += normals.points.at(pi).normal[2];
// 	    
// 	      left_point.x += point_x;
// 	      left_point.y += point_y;
// 	      left_point.z += point_z;
// 	      
// // 	      pcl::PointXYZ temp_left;
// // 	      temp_left.x = point_x;
// // 	      temp_left.y = point_y;
// // 	      temp_left.z = point_z;
// 	      
// 	      //points_left->points.push_back(temp_left);
// 	      //normals_left->points.push_back(normals.points.at(pi));
// 	    }
// 	    else if(dist_to_the_plane < 0)
// 	    {
// 	      right_num += 1;
// 	      right_distance += distances.at(pi);
// 		   
// 	      right_normal.normal[0] += normals.points.at(pi).normal[0];
// 	      right_normal.normal[1] += normals.points.at(pi).normal[1];
// 	      right_normal.normal[2] += normals.points.at(pi).normal[2];
// 	   
// 	      right_point.x += point_x;
// 	      right_point.y += point_y;
// 	      right_point.z += point_z;
// 	      
// // 	      pcl::PointXYZ temp_right;
// // 	      temp_right.x = point_x;
// // 	      temp_right.y = point_y;
// // 	      temp_right.z = point_z;
// 	      
// 	      //points_right->points.push_back(temp_right);
// 	      //normals_right->points.push_back(normals.points.at(pi));
// 	    }
//           }
//         
// 	
// 	
// 	  if((left_num > 0) && (right_num > 0))
// 	  {
// 	    left_distance /= left_num;
// 	    right_distance /= right_num;
// 	
// 	    float Wi = right_distance - left_distance;
// 	    Wi = (Wi > 0 ? Wi : -Wi);
// 	    
// 	    left_normal.normal[0] /= left_num;
// 	    left_normal.normal[1] /= left_num;
// 	    left_normal.normal[2] /= left_num;
// 	    
// 	    right_normal.normal[0] /= right_num;
// 	    right_normal.normal[1] /= right_num;
// 	    right_normal.normal[2] /= right_num;
// 	    
// 	    left_point.x /= left_num;
// 	    left_point.y /= left_num;
// 	    left_point.z /= left_num;
// 	    
// 	    right_point.x /= right_num;
// 	    right_point.y /= right_num;
// 	    right_point.z /= right_num;
// 	    
// 	/*points_left->points.push_back(left_point);
// 	normals_left->points.push_back(left_normal);
// 	points_right->points.push_back(right_point);
// 	normals_right->points.push_back(right_normal);
// 	
// 	pcl::PointCloud<pcl::PointXYZ>::Ptr small_cloud_XYZ (new pcl::PointCloud<pcl::PointXYZ>());
// 	EPUtils::convertPointXYZRGBRC2PointXYZ(small_cloud.makeShared(),small_cloud_XYZ);
// 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//         viewer->setBackgroundColor (0, 0, 0);
// 	viewer->addPointCloud<pcl::PointXYZ> (small_cloud_XYZ, "small_cloud_XYZ");
//         viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "small_cloud_XYZ");
//         // left
// 	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> left_color(points_left, 0, 0, 255);
// 	viewer->addPointCloud<pcl::PointXYZ> (points_left, left_color, "points_left");
//         viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "points_left");
//         viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (points_left, normals_left, 10, 0.02, "normals_left");
// 	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "normals_left");
// 	// right_distance
// 	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> right_color(points_right, 255, 255, 0);
// 	viewer->addPointCloud<pcl::PointXYZ> (points_right, right_color, "points_right");
//         viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "points_right");
//         viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (points_right, normals_right, 10, 0.02, "normals_right");
// 	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "normals_right");
// 	
// 	//viewer->addPlane(coefficients);
// 	while (!viewer->wasStopped ())
//         {
//           viewer->spinOnce (100);
//           boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//         }*/
// 	    
// 	    pcl::Normal lineNormal;
// 	    lineNormal.normal[0] = left_point.x - right_point.x;
// 	    lineNormal.normal[1] = left_point.y - right_point.y;
// 	    lineNormal.normal[2] = left_point.z - right_point.z;
// 	    EPUtils::normalizeVector(lineNormal);
// 	    
// 	    pcl::Normal N;
// 	    EPUtils::calculatePlaneNormal(lineNormal,right_normal,N);
// 	    float Ci;
// 	    EPUtils::calculateCosine(left_normal,N,Ci);
// 	    Ci = Ci > 0 ? Ci : -Ci;
// 	    Ci = sqrt(1-Ci*Ci);
// 	    
// 	    float d=plane_normal.normal[0]*(left_point.x-x0)+plane_normal.normal[1]*(left_point.y-y0)+plane_normal.normal[2]*(left_point.z-z0);
// 	    float cos_left=0, cos_right=0;
// 	    EPUtils::calculateCosine(left_normal,plane_normal,cos_left);
// 	    EPUtils::calculateCosine(right_normal,plane_normal,cos_right);
// 	    bool point_is_ok=true;
// 	    if (d<0)
// 	    {
// 		// cos_left > 90deg && cos_right < 90deg
// 		point_is_ok = (cos_left<0) && (cos_right>0); 
// 	    }
// 	    else 
// 	    {
// 		// cos_left < 90deg && cos_right > 90deg
// 		point_is_ok = (cos_left>0) && (cos_right<0);
// 	    }
// 	     
// 	    float cos1, cos2;
// 	    EPUtils::calculateCosine(lineNormal,left_normal,cos1);
// 	    EPUtils::calculateCosine(lineNormal,right_normal,cos2);
// 	    //cos2 = -cos2;
// 	    
// 	    float alpha1 = acos(cos1);
// 	    float alpha2 = acos(cos2);
// 	    
// 	    float Si;
// 	    Si = (1-cos(alpha1+alpha2))*(1-cos(alpha1-alpha2));
// 	    
// 	    float Di = right_point.z - left_point.z;
// 	    Di = Di > 0 ? Di : -Di;
// 	    
// 	    if((left_point.z > 0) && (right_point.z > 0) && (right_distance > 0) && (left_distance > 0) && point_is_ok)
// 	    {
// 	      /*std::cerr << "Di = " << Di << std::endl;
// 	      std::cerr << "Wi = " << Wi << std::endl;
// 	      std::cerr << "alpha1 = " << alpha1 << std::endl;
// 	      std::cerr << "alpha2 = " << alpha2 << std::endl;
// 	      std::cerr << "Si = " << Si << std::endl;
// 	      std::cerr << "Ci = " << Ci << std::endl;
// 	      std::cerr << "axis_num = " << axis_num << std::endl;
// 	      std::cerr << "cheat = " << cheat << std::endl;*/
// 	      W.at(axis_num) = exp(-1000*Wi)*exp(-1000*Di)*Si*Ci;
// 	    }
// 	    
// 	    /*boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//             viewer->setBackgroundColor (0, 0, 0);
//             viewer->addPointCloud<pcl::PointXYZ> (small_cloud_XYZ, "small_cloud_XYZ");
//             viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "small_cloud_XYZ");
//             viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (small_cloud_XYZ, normals.makeShared(), 1, 0.05, "normals");
//             viewer->addCoordinateSystem (1.0);
// 	    viewer->addPlane(coefficients);
// 	    while (!viewer->wasStopped ())
//             {
//               viewer->spinOnce (100);
//               boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//             }*/
// 	    
// 	  }
// 	}
// 	
// 	// calculate number of valid principle planes
// 	float W_max = 0;
// 	for(unsigned int axis_num = 0; axis_num < axis.size(); ++axis_num)
// 	{
// 	   if(W.at(axis_num) > W_max)
// 	   {
// 	      W_max = W.at(axis_num);
// 	   }
// 	}
// 	  
// 	parameters.map.at<float>(rr,cc) = W_max;
// 	
//       }
//     }
//   }
//   
//   int filter_size = parameters.filter_size;
//   cv::blur(parameters.map,parameters.map,cv::Size(filter_size,filter_size));
//   
//   //EPUtils::normalize(parameters.map,parameters.normalization_type);
//   double minVal, maxVal;
//   cv::minMaxLoc(parameters.map,&minVal,&maxVal);
//   parameters.map = (1.0/maxVal)*parameters.map;
// }
// 
// void Calculate3DSymmetryPCAPyramid(AttentionModule::Symmetry3DMapParameters &parameters)
// {
//   // create depth
//   cv::Mat depth;
//   EPUtils::convertPointCloud2Depth(parameters.cloud,parameters.indices,depth,parameters.width,parameters.height);
//   
//   parameters.normalization_type = EPUtils::NT_NONMAX;
//   parameters.pyramidParameters.start_level = 0;
//   parameters.pyramidParameters.max_level = 2;
//   
//   parameters.cameraParametrs.clear();
//   parameters.cameraParametrs.resize(4);
//   parameters.cameraParametrs.at(0) = 525.0f;
//   parameters.cameraParametrs.at(1) = 525.0f;
//   parameters.cameraParametrs.at(2) = 319.5f;
//   parameters.cameraParametrs.at(3) = 239.5f;
//   
//   parameters.pyramidParameters.combination_type = AttentionModule::AM_SIMPLE;
//   parameters.pyramidParameters.normalization_type = EPUtils::NT_NONMAX;
//   
//   // calculate puramid with saliency maps
//   int max_level = parameters.pyramidParameters.max_level + 1;
//   parameters.pyramidParameters.pyramidImages.clear();
//   cv::buildPyramid(depth,parameters.pyramidParameters.pyramidImages,max_level);
//   parameters.pyramidParameters.pyramidFeatures.clear();
//   parameters.pyramidParameters.pyramidFeatures.resize(parameters.pyramidParameters.pyramidImages.size());
//   
//   //parameters.pyramidParameters.print();
//   
//   for(int i = parameters.pyramidParameters.start_level; i <= parameters.pyramidParameters.max_level; ++i)
//   {
//     std::cerr << "level = " << i << std::endl; 
//     
//     int scalingFactor = pow(2.0f,i);
//     std::vector<float> cameraParametrs;
//     cameraParametrs.resize(4);
//     cameraParametrs.at(0) = parameters.cameraParametrs.at(0)/scalingFactor;
//     cameraParametrs.at(1) = parameters.cameraParametrs.at(1)/scalingFactor;
//     cameraParametrs.at(2) = parameters.cameraParametrs.at(2)/scalingFactor;
//     cameraParametrs.at(3) = parameters.cameraParametrs.at(3)/scalingFactor;
//     
//     // start creating parameters
//     AttentionModule::Symmetry3DMapParameters parameters_current;
//     parameters_current.width = parameters.pyramidParameters.pyramidImages.at(i).cols;
//     parameters_current.height = parameters.pyramidParameters.pyramidImages.at(i).rows;
//     
//     // create scaled point cloud
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
//     EPUtils::convertDepth2PointCloud(parameters.pyramidParameters.pyramidImages.at(i),cloud,cameraParametrs);
//     parameters_current.cloud = cloud;
//     
//     pcl::PointIndices::Ptr indices(new pcl::PointIndices());
//     indices->indices.resize(cloud->points.size());
//     for(unsigned int idx = 0; idx < cloud->points.size(); ++idx)
//     {
//       indices->indices.at(idx) = idx;
//     }
//     
//     //calculate point cloud normals
//     if(!pclAddOns::ComputePointNormals<pcl::PointXYZ>(parameters_current.cloud,indices,parameters_current.normals))
//       return;//(AttentionModule::AM_NORMALCLOUD);
//     
//     //CalculateSymmetry3DMap(parameters_current);
//     Calculate3DSymmetryPCA(parameters_current);
//     parameters_current.map.copyTo(parameters.pyramidParameters.pyramidFeatures.at(i));
//     parameters_current.map.copyTo(parameters.pyramidParameters.pyramidImages.at(i));
//   }
// 
//   // combine saliency maps
//   combinePyramid(parameters.pyramidParameters);
//   parameters.pyramidParameters.map.copyTo(parameters.map);
// }
// 
// boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis (
//     pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
// {
//   // --------------------------------------------------------
//   // -----Open 3D viewer and add point cloud and normals-----
//   // --------------------------------------------------------
//   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//   viewer->setBackgroundColor (0, 0, 0);
//   viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
//   viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
//   viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, normals, 1, 0.05, "normals");
//   viewer->addCoordinateSystem (1.0);
//   viewer->initCameraParameters ();
//   return (viewer);
// }
// 
// }
// 
// // int main(int argc, char** argv)
// // {
// // //   std::string directory_images("/home/kate/work/database/TOSD/TOSD/learn/rgb/");
// // //   std::string directory_clouds("/home/kate/work/database/TOSD/TOSD/learn/clouds/");
// // //   std::string directory_sms("/home/kate/work/database/TOSD/TOSD/learn/r/");
// //   
// // //   std::vector<std::string> images_names;
// // //   std::vector<std::string> clouds_names;
// //   
// // //   EPUtils::readFiles(directory_images,images_names);
// // //   EPUtils::readFiles(directory_clouds,clouds_names);
// //   
// // //   for(int i = 0; i < images_names.size(); ++i)
// //   {
// // //     std::cerr << "Image: " << images_names.at(i) << std::endl;
// // //     std::cerr << "Cloud: " << clouds_names.at(i) << std::endl;
// //   
// //     // read image
// //     std::string image_name(argv[1]);
// //     cv::Mat image = cv::imread(image_name,-1);
// // //     cv::Mat image = cv::imread(images_names.at(i),-1);
// //     
// //     pcl::PointCloud<pcl::PointXYZ> temp_cloud;
// //     std::string cloud_name(argv[2]);
// //     if (pcl::io::loadPCDFile<pcl::PointXYZ> (/*clouds_names.at(i)*/cloud_name,temp_cloud) == -1)
// //     {
// //       std::cerr << "[ERROR] Couldn't read point cloud." << std::endl;
// //       return -1;
// //     }
// //   
// //     // start creating parameters
// //     AttentionModule::Symmetry3DMapParameters parameters;
// //     parameters.width = image.cols;
// //     parameters.height = image.rows;
// //     
// //     // create scaled point cloud
// //     pcl::PointCloud<pclAddOns::PointXYZRGBRC>::Ptr cloud;
// //     EPUtils::convertPointXYZ2PointXYZRGBRC(temp_cloud.makeShared(),/*cloud*/parameters.cloud,parameters.width,parameters.height);
// //   
// //     //filter just obtained point cloud
// // //     if(!pclAddOns::FilterPointCloud<pclAddOns::PointXYZRGBRC>(cloud,parameters.cloud))
// // //       return(pclAddOns::FILTER);
// //     
// //     //calculate point cloud normals
// //     if(!pclAddOns::ComputePointNormals<pclAddOns::PointXYZRGBRC>(parameters.cloud,parameters.normals,50))
// //       return(pclAddOns::NORMALS);
// //   
// // //     Calculate3DSymmetryPCAPyramid(parameters);
// //     Calculate3DSymmetryPCA(parameters);
// //     
// //     cv::Mat symmetry_map;
// //     parameters.map.convertTo(symmetry_map,CV_8U,255);
// //     
// // //     std::string map_path = directory_sms;
// // //     map_path += boost::filesystem::basename(images_names.at(i)) + ".pgm";
// //     
// // //     std::cerr << "writing image:" << map_path << std::endl;
// //     
// // //     cv::imwrite(map_path,symmetry_map);
// //     cv::imwrite("sphere_sm.png",symmetry_map);
// //   
// //   /*pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ (new pcl::PointCloud<pcl::PointXYZ>);
// //   EPUtils::convertPointXYZRGBRC2PointXYZ(parameters.cloud,cloudXYZ);
// //   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
// //   viewer = normalsVis(cloudXYZ,parameters.normals);
// //   while (!viewer->wasStopped ())
// //   {
// //     viewer->spinOnce (100);
// //     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
// //   }*/
// // 
// //   //cv::imshow("map",parameters.map);
// //   //cv::waitKey();
// //   }
// //     
// //   return(0);
// // }