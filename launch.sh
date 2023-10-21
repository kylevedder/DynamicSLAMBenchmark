#!/bin/bash
touch docker_history.txt
xhost +
docker run --gpus=all --rm -it \
 --shm-size=16gb \
 -v `pwd`:/scene_trajectory_benchmark \
 -v /efs:/efs \
 -v /bigdata:/bigdata \
 -v /tmp:/tmp \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v /tmp/frame_results:/tmp/frame_results \
 -v `pwd`/docker_history.txt:/root/.bash_history \
 -e DISPLAY=$DISPLAY \
 -h $HOSTNAME \
 --privileged \
 scene_trajectory_benchmark:latest
