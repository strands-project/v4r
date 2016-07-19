The library itself is independent of ROS, so it is built outside ROS catkin. There are wrappers for ROS (https://github.com/strands-project/v4r_ros_wrappers), which can then be placed inside the normal catkin workspace.

# Dependencies:  
stated in [`package.xml`](https://github.com/strands-project/v4r/blob/master/package.xml)
There are two options to use the SIFT recognizer:
 - Use V4R third party library SIFT GPU (this requires a decent GPU - see www.cs.unc.edu/~ccwu/siftgpu) [default]
 - Use OpenCV non-free SIFT implementation (this requires the non-free module of OpenCV - can be installed from source). This option is enabled if BUILD_SIFTGPU is disabled in cmake.

# Installation:  
In order to use V4R in ROS, use the [v4r_ros_wrappers](https://github.com/strands-project/v4r_ros_wrappers/blob/master/Readme.md).

## From Ubuntu Package  
simply install `sudo apt-get install ros-indigo-v4r` after enabling the [STRANDS repositories](https://github.com/strands-project-releases/strands-releases/wiki#using-the-strands-repository).

## From Source  
```
cd ~/somewhere
git clone 'https://github.com/strands-project/v4r'
cd v4r
mkdir build
cd build
cmake ..
make
sudo make install (optional)
```
