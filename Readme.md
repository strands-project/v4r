The library itself is independent of ROS, so it is built outside ROS catkin. There are wrappers for ROS (https://github.com/strands-project/v4r_ros_wrappers), which can then be placed inside the normal catkin workspace.

Dependencies:
-------------
stated in package.xml

Installation:
-------------
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

