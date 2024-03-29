#!/bin/bash
image_name=${1:-cobot_vision}
container_name=${2:-cobot_vision_yangfei}

docker container stop $container_name && docker container prune -f
xhost +local:docker 
echo "starting pvnet docker"

docker run -it \
  --name $container_name \
  --volume=$PWD:/pvnet \
  --privileged=true\
  --env="XAUTHORITY=$XAUTH" \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --runtime=nvidia \
  --net=host --gpus all \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --shm-size=16G \
  --device=/dev/dri:/dev/dri \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /etc/localtime:/etc/localtime:ro \
  -v /data/models:/data/models \
  $image_name bash 
