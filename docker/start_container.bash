#!/usr/bin/bash
container_name=${1:-cobot_vision}
xhost +local:docker 
docker start $container_name && docker exec -it \
  -e DISPLAY=$DISPLAY \
  --privileged \
  $container_name bash
