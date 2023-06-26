# README

## Build docker image 

```bash
docker build -t pvnet_clean:latest .
```

## Build docker container

To run the docker
Add the following to your ~/.bashrc

```bash
export PVNET_DOCKER=pvnet_clean:latest
export PVNET_GIT=$HOME/clean-pvnet  # update
source $PVNET_GIT/docker/setup_dev.bash
```


build a docker container and mount it with your local folder:
```bash
cd /path/to/clean-pvnet # update
bash docker/build_container.bash
```

## Start docker container
start docker container:
```bash
# If it's your first time to build the container, no need to run this line:
bash docker/start_container.bash
cd pvnet
conda activate pvnet
```

Test:
```bash
python run.py --type linemod cls_type cat
python run.py --type visualize --cfg_file configs/linemod.yaml model cat cls_type cat
```