#!/usr/bin/bash
container_name=${1:-cobot_vision}
xhost +local:root # Allow connections to X server
docker start $container_name && docker exec -it \
  -e DISPLAY=$DISPLAY \
  --privileged \
  $container_name bash
xhost -local:root # Reset X server access permissions