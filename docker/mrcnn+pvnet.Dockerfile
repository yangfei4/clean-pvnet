# This image was constrcuted following instructions outlined: http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration
# Please refer to the resources above
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
RUN apt-get install -y  python-pip python3-pip git

RUN apt install -y python3-dev \
	&& update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \ 
	&& python -m pip install --upgrade pip \
	&& python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
	&& python -m pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6" \
	&& python -m pip install rospkg catkin_pkg opencv-python==4.5.5.64 gin-config empy==3.3.2 scipy transforms3d

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


# installing PVnet dependencies (and removing pvnet again)
ENV PIP_ROOT_USER_ACTION=ignore

RUN python -m pip install Cython==0.28.2 && \
    python -m pip install \
    numpy  \
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
    # USE Detectron2's default installs instead
    #yacs==0.1.4 \ Use D
    #torchvision==0.2.1 \


RUN python -m pip install transforms3d pillow==8.1.0 ninja
CMD ["/bin/bash"]



#######################################
# ROS Related Section
#######################################
# Set up the ROS repository
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y \
    curl gnupg2 lsb-release \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-get update
# Install ROS Melodic
RUN apt-get install -y ros-melodic-desktop-full
# Install additional ROS dependencies
RUN apt-get install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
# Initialize rosdep
RUN rosdep init && rosdep update
# Set up ROS environment

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
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" python setup.py build_ext --inplace #&& \

RUN apt update && apt install -y python3-catkin-pkg-modules python3-rospkg-modules python3-empy wget \
    && source /opt/ros/melodic/setup.bash \
    && bash $ROOT/src/build_ros_ws \
    && echo "source ${ROOT}/devel/setup.bash" >> ~/.bashrc
    # rm -rf $ROOT 

RUN source /opt/ros/melodic/setup.bash  && apt install  -y tmux ranger && echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc \
    && source ~/.bashrc

RUN apt-get update && apt-get install -y \
      python-catkin-tools \
    && rm -rf /var/lib/apt/lists/*
RUN pip install netifaces

WORKDiR /pvnet
# COPY /src /pvnet/src
# NOTE reference for building pyton 3 containers: https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/
# RUN apt update && apt install -y python3-catkin-pkg-modules python3-rospkg-modules python3-empy wget \
#     && source /opt/ros/melodic/setup.bash \
#     && bash src/build_ros_ws \
#     && echo "source /pvnet/devel/setup.bash" >> ~/.bashrc


# Set the container's main command
#CMD ["bash"]
