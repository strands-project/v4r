#!/bin/bash
echo "Installing Dependencies for V4R (Ubuntu 14.04 Trusty)..."

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo apt-get update -qq
sudo apt-get install -qq -y python-rosdep build-essential cmake
sudo rosdep init

rosdep update
rosdep install --from-paths . -i -y -r --rosdistro indigo #maybe 'indigo' has to be changed for Ubuntu 16.04 to 'jade' or 'kinetic'

echo "Building V4R..."
cpu=`nproc`
cpu=$(($cpu -1)) #extracting num. cpu_cores-1
mkdir build
cd build
cmake ..
make -j$cpu
