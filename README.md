# PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation
<p align="center">
  <img src="https://github.com/yangfei4/Sim2real/blob/main/figures/pvnet_pipeline.jpg?raw=true" width="800">
</p>
<p align="center">
  <img src="https://github.com/yangfei4/Sim2real/blob/main/figures/pvnet_challenges.jpg?raw=true" width="800">
</p>

#### Evaluation Results [Evaluation Pipeline](https://github.com/yangfei4/clean-pvnet/blob/master/evaluation.ipynb)
<p align="center">
  <img src="https://github.com/yangfei4/Sim2real/blob/main/figures/pvnet_evaluation.jpg?raw=true" width="800">
</p>

## Installation

One way is to set up the environment with docker. See [this](https://github.com/yangfei4/clean-pvnet/tree/master/docker).

Another way is to use the following commands.
<details>
<summary> Click me to expand ...</summary>
1. Set up the python environment:
    ```
    conda create -n pvnet python=3.7
    conda activate pvnet

    # install torch 1.1 built from cuda 9.0
    pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable

    pip install Cython==0.28.2
    sudo apt-get install libglfw3-dev libglfw3
    pip install -r requirements.txt
    ```
2. Compile cuda extensions under `lib/csrc`:
    ```
    ROOT=/path/to/clean-pvnet
    cd $ROOT/lib/csrc
    export CUDA_HOME="/usr/local/cuda-9.0"
    cd ransac_voting
    python setup.py build_ext --inplace
    cd ../nn
    python setup.py build_ext --inplace
    cd ../fps
    python setup.py build_ext --inplace
    
    # If you want to run PVNet with a detector
    cd ../dcn_v2
    python setup.py build_ext --inplace

    # If you want to use the uncertainty-driven PnP
    cd ../uncertainty_pnp
    sudo apt-get install libgoogle-glog-dev
    sudo apt-get install libsuitesparse-dev
    sudo apt-get install libatlas-base-dev
    python setup.py build_ext --inplace
    ```
3. Set up datasets:
    ```
    ROOT=/path/to/clean-pvnet
    cd $ROOT/data
    ln -s /path/to/linemod linemod
    ln -s /path/to/linemod_orig linemod_orig
    ln -s /path/to/occlusion_linemod occlusion_linemod

    # the following is used for tless
    ln -s /path/to/tless tless
    ln -s /path/to/cache cache
    ln -s /path/to/SUN2012pascalformat sun
    ```
</details>

## Training on the custom object

The training parameters can be found in [project_structure.md](project_structure.md).

1. Create a synthetic dataset with mask and pose annotation using Blenderproc2. 
2. Organize the dataset as the following structure:
    ```
    ├── /path/to/dataset
    │   ├── model.ply
    │   ├── camera.txt
    │   ├── diameter.txt  // the object diameter, whose unit is meter
    │   ├── rgb/
    │   │   ├── 0.jpg
    │   │   ├── ...
    │   │   ├── 1234.jpg
    │   │   ├── ...
    │   ├── mask/
    │   │   ├── 0.png
    │   │   ├── ...
    │   │   ├── 1234.png
    │   │   ├── ...
    │   ├── pose/
    │   │   ├── pose0.npy
    │   │   ├── ...
    │   │   ├── pose1234.npy
    │   │   ├── ...
    │   │   └──
    ```
2.  Create a soft link pointing to the training and testing dataset:
    ```
    ln -s /path/to/dataset data/category
    ```
    insert_mold:
    ```
    ln -s /pvnet/data/FIT/insert_mold_train data/insert_mold_train
    ln -s /pvnet/data/FIT/insert_mold_test data/insert_mold_test
    ```
    And remember to modify customized soft path for you customized dataset [here](https://github.com/yangfei4/clean-pvnet/blob/master/lib/datasets/dataset_catalog.py).
3.  Process the dataset, this will create `train.json` and `fps.txt`:
    ```
    python run.py --type insert_mold
    python run.py --type insert_mold_test
    ```
4. Visualize Pose Ground Truth of training dataset
    ```
    python run.py --type visualize_gt --cfg_file configs/insert_mold.yaml
    ```
5. Train:
    ```
    python train_net.py --cfg_file configs/insert_mold.yaml train.batch_size 32
    ```
6. Watch the training curve:
    ```
    tensorboard --logdir data/record/pvnet
    ```
7. Visualize:
    ```
    python run.py --type visualize --cfg_file configs/insert_mold.yaml
    python run.py --type visualize_train --cfg_file configs/insert_mold.yaml
    ```
8. Test:
    ```
    python run.py --type evaluate --cfg_file configs/insert_mold.yaml
    or
    python run.py --type evaluate --cfg_file configs/insert_mold.yaml test.un_pnp True
    ```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{peng2019pvnet,
  title={PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation},
  author={Peng, Sida and Liu, Yuan and Huang, Qixing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={CVPR},
  year={2019}
}
```

## Acknowledgement

This work is affliated with ZJU-SenseTime Joint Lab of 3D Vision, and its intellectual property belongs to SenseTime Group Ltd.

```
Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
