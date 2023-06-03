# README

## Build 

```bash
docker build -t pvnet_clean:latest .
```

## Run

To run the docker
Add the following to your ~/.bashrc

```bash
export PVNET_DOCKER=pvnet_clean:latest
export PVNET_GIT=$HOME/gits/clean-pvnet  # update
source $PVNET_GIT/docker/setup_dev.bash
```

To make your GPU available for this docker environment:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

build a docker container and mount it with your local folder:
```bash
#please updata path to local folder
docker run -it --name pvnet --gpus all -v /path/to/clean-pvnet:$PWD:/pvnet pvnet_clean 
```

```bash
docker start pvnet && docker exec -it pvnet bash
cd pvnet
```
