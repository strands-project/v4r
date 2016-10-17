#!/bin/bash

if [ $# -eq 0 ]
  then
    # No arguments supplied
    ubuntu_version="14.04"
    ubuntu_version_name="trusty"
    ros_version="indigo"

elif [ $1 = "xenial" ]
  then
    ubuntu_version="16.04"
    ubuntu_version_name="xenial"
    ros_version="kinetic"
fi

echo "Installing Dependencies for V4R (Ubuntu ${ubuntu_version} ${ubuntu_version_name} using ROS ${ros_version})..."
echo "If you want to change this you can pass in the codename of the Ubuntu release. Eg. $0 xenial for 16.04"

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu '${ubuntu_version_name}' main" > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo apt-get update -qq
sudo apt-get install -qq -y python-rosdep build-essential cmake
sudo rosdep init

rosdep update
rosdep install --from-paths . -i -y -r --rosdistro ${ros_version} 
