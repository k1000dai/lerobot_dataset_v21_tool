#!/bin/bash

# Setup sources
apt update && apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package index
apt update

# Install minimal packages (replace 'humble' with your ROS2 distro: iron, jazzy, etc.)
apt install -y \
    ros-humble-tf2-ros \
    ros-humble-tf2-ros-py \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs

source /opt/ros/humble/setup.bash
