#!/bin/bash

for TESTFOR in "NVIDIA-Linux-x86_64-367.57.run" "nvidia-docker_1.0.0.rc.3-1_amd64.deb" 
do
    if [ ! -e $TESTFOR ]
    then
        echo File '"'$TESTFOR'"' must be present in this directory
        exit 5
    fi
done

# Reference:
#   https://github.com/NVIDIA/nvidia-docker/wiki/Deploy-on-Amazon-EC2

# Install NVIDIA drivers 367.57
#  This is incrementally newer than the version that we get updated to below (367.48) if we follow the instructions and install 361.42
#  Hoping this works since 367.48 has been removed from the NVidia downloads site
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
sudo apt-get -q install --no-install-recommends -y gcc make libc-dev
sudo sh NVIDIA-Linux-x86_64-367.57.run --ui=none --no-questions --accept-license
