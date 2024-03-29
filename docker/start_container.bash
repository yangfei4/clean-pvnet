#!/usr/bin/bash
container_name=${1:-cobot_vision_yangfei}
xhost +local:docker 
docker start $container_name && docker exec -it \
  -e DISPLAY=$DISPLAY \
  --privileged \
  $container_name bash
