# This image was constrcuted following instructions outlined: http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration
# Please refer to the resources above
# FROM yangfei4/pvnet_clean:ros
#FROM chapchaebytes/ur5e_collab_ws
# FROM osrf/ros:melodic-desktop-full
FROM nvidia/cuda:11.1.1-devel-ubuntu18.04
SHELL ["/bin/bash", "-c"]

# Hack to not have tzdata cmdline config during build
RUN ln -fs /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
     ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
     ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compute,utility

# Install pytorch 1.10.0 for CUDA 11.1 (https://pytorch.org/get-started/previous-versions/#linux-and-windows-13)
# Install Detectron2 v0.6
RUN apt update
# RUN apt-get install -y software-properties-common python-pip python3-pip ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-moveit ros-melodic-trac-ik-kinematics-plugin && add-apt-repository ppa:sdurobotics/ur-rtde && apt-get update && apt install librtde librtde-dev
RUN apt-get install -y  python-pip python3-pip git

RUN apt install -y python3-dev \
	&& update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \ 
	&& python -m pip install --upgrade pip \
	#&& python -m pip install --no-cache-dir torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
	#&& python -m pip install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html \
	&& python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
	&& python -m pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6" \
	&& python -m pip install rospkg catkin_pkg opencv-python==4.5.5.64 gin-config empy scipy transforms3d

RUN python -m pip uninstall yacs -y \
    && python -m pip install yacs --upgrade 

#######################################
# PVnet Related Section
#######################################
RUN apt-get update && \
    apt install -yq software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -yq \
        nano \
        sudo \
        wget \
        curl \
        build-essential \
        cmake \
        git \
        ca-certificates

# Update package lists and install necessary packages
RUN apt-get update \
    && apt-get install -yq  libssl-dev zlib1g-dev libbz2-dev \
       libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
       xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

# Install pyenv
# ENV PYENV_ROOT=/root/.pyenv
# ENV PATH=/root/.pyenv/bin:$PATH
# RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# Set up pyenv
# RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc \
#     && echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc \
#     && echo 'export PATH="/root/.pyenv/bin:$PATH"' >> ~/.bashrc \
#     && echo 'eval "$(pyenv init -)"' >> ~/.bashrc
# 
# # Install Python 3.7.12
# RUN pyenv install 3.7.12 \
#     && pyenv global 3.7.12
# 
# # Verify Python installation
# RUN python --version

# RUN apt install -yq python3.7 \
        # python3-pip \

RUN apt install -yq python-qt4 \
        libjpeg-dev \
        zip \
        unzip \
        libpng-dev \
        libeigen3-dev \
        libglfw3-dev \
        libglfw3 \
        libgoogle-glog-dev \
        libsuitesparse-dev \
        libatlas-base-dev \
        python3-tk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# (mini)conda
# https://repo.anaconda.com/miniconda/
# RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
#     sh ./Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -p /opt/conda && \
#     rm ./Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
#     export PATH=$PATH:/opt/conda/bin && \
#     conda install conda-build
# 
# ENV PATH $PATH:/opt/conda/envs/env/bin:/opt/conda/bin

# installing PVnet dependencies (and removing pvnet again)
ENV PIP_ROOT_USER_ACTION=ignore

# RUN cd /opt && \
#     git clone https://github.com/yangfei4/clean-pvnet.git pvnet && \
#     cd pvnet && \
#     cat requirements.txt | xargs -n 1 python -m pip install

RUN python -m pip install Cython==0.28.2 && \
    python -m pip install \
    yacs==0.1.4 \
    numpy  \
    torchvision==0.2.1 \
    opencv-python==3.4.2.17 \
    tqdm==4.28.1 \
    pycocotools \
    matplotlib==2.2.2 \
    plyfile==0.6 \
    scikit-image==0.14.2 \
    scikit-learn \
    PyOpenGL==3.1.1a1 \
    ipdb \
    cyglfw3==3.1.0.2 \
    pyassimp==3.3 \
    progressbar==2.5 \
    open3d-python==0.5.0.0 \
    tensorboardX==1.2 \
    cffi==1.11.5

RUN python -m pip install transforms3d pillow==8.1.0 ninja

ENV ROOT=/opt/pvnet
ENV CUDA_HOME="/usr/local/cuda"

# NOTE: Be sure that the arch version is compatible with your GPU/CUDA version
# See: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
RUN cd /opt && \
    git clone https://github.com/yangfei4/clean-pvnet.git pvnet && \cd $ROOT/lib/csrc && \
    cd ransac_voting && \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" python setup.py build_ext --inplace && \
    cd ../nn && \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" python setup.py build_ext --inplace && \
    cd ../fps && \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" python setup.py build_ext --inplace && \
    rm -rf $ROOT 

CMD ["/bin/bash"]



#######################################
# ROS Related Section
#######################################
# WORKDIR  /cobot_detectron2
# COPY src /cobot_detectron2/src
# 
# RUN apt install -y udev 
# RUN sh -c 'echo "yaml https://raw.githubusercontent.com/basler/pylon-ros-camera/master/pylon_camera/rosdep/pylon_sdk.yaml" > /etc/ros/rosdep/sources.list.d/30-pylon_camera.list' \
# RUN source /opt/ros/melodic/setup.bash && apt-get install -y apt-utils && apt update -qq && sudo rosdep fix-permissions --rosdistro=melodic && rosdep update --rosdistro=melodic \
#     && rosdep install --from-paths src --ignore-src --rosdistro=melodic -y
# 
# 
# RUN source /opt/ros/melodic/setup.bash  && apt install  -y tmux && echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# 
# RUN apt-get update && apt-get install -y \
#       python-catkin-tools \
#     && rm -rf /var/lib/apt/lists/*
# 
# # NOTE reference for building pyton 3 containers: https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/
# RUN apt update && apt install -y python3-catkin-pkg-modules python3-rospkg-modules python3-empy wget \
#     && source /opt/ros/melodic/setup.bash \
# 	&& catkin_make --cmake-args \
#             -DCMAKE_BUILD_TYPE=Release \
#             -DPYTHON_EXECUTABLE=/usr/bin/python3 \
#             -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
#             -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
# 	&& echo "source devel/setup.bash" >> ~/.bashrc
