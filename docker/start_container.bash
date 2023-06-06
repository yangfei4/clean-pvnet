#!/usr/bin/bash
container_name=${1:-pvnet}
docker start $container_name && docker exec -it $container_name bash