# README

## Build Docker Image
```bash
sudo bash docker/build_image
```

## Build Docker Container
```bash
sudo bash docker/build_container.bash
```

## (If the docker container has already been built) Start docker container
start docker container:
```bash
bash docker/start_container.bash
```

Test:
```bash
python run.py --type linemod cls_type cat
python run.py --type visualize --cfg_file configs/linemod.yaml model cat cls_type cat
```