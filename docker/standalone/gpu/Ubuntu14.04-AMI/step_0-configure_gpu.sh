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

sudo apt-get -qq update -y
sudo apt-get -q install --no-install-recommends -y gcc make libc-dev wget

# Install NVIDIA drivers 367.57
#  This is incrementally newer than the version 367.48 that the Ubuntu repos hold.
#  NVidia no longer has the 367.48 version on their downloads site and we have
#  to have the *.run file so that we match the host and the Docker container
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
sudo sh NVIDIA-Linux-x86_64-367.57.run --ui=none --no-questions --accept-license
