this is intended to contain ROS wrappers to some v4r and faat_framework components. Right now, the following things are wrapped:

 * Helpers
	* PACKAGE NAME: segmentation_msgs_and_service
		* messages and service definition for segmenters
	
 * V4R wrappers
	* segmentation from Kate/Andreas
		* PACKAGE NAME: object_rgbd_segmenter

 * faat_framework wrappers
 	* object instance recognition and pose estimation (single view)
 		* PACKAGE NAME: sv_obj_instance_recognition
 		
 * PCL wrappers
 	* PCL based segmentation (dominant plane assumption or multiple plane segmentation)
 		* PACKAGE NAME: pcl_object_segmenter
  	
Compilation: 	
catkin_make can be called as follows:

catkin_make -DPCL_DIR=/home/aitor/pcl_git_fork/pcl/build/ -DFAAT_PCL_DIR=/home/aitor/v4r/faat_build/ -DV4R_DIR=/home/aitor/projects/strandsv4r/

remember to specify the paths properly to point to the PCL location with PCLConfig.cmake.in and the same for FAAT_PCL_DIR.

TODO:

	* segmenters should accept an up-right vector to segment only horizontal planes
	* organized component based pcl segmenter should segment multiple planes
		* what should be the table plane returned if any?
	* add classification wrapper
		* scale (learn scale distributions based on labeled data from KTH)
		* pose estimation similar to GRASP project
	
	* multiview classification
		* registration module needed
	* ROS wrapper for object reconstruction?
