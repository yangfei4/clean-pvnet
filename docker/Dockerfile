FROM nvidia/cuda:9.0-devel-ubuntu16.04

# Hack to not have tzdata cmdline config during build
RUN ln -fs /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install python3.7 and dependencies, taken from:
# - hhttps://websiteforstudents.com/installing-the-latest-python-3-7-on-ubuntu-16-04-18-04/
# - https://github.com/zju3dv/pvnet/blob/master/docker/Dockerfile
# - https://github.com/zju3dv/clean-pvnet

# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys <full_key_id>
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC


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
    && apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
       libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
       xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

# Install pyenv
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=/root/.pyenv/bin:$PATH
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# Set up pyenv
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc \
    && echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc \
    && echo 'export PATH="/root/.pyenv/bin:$PATH"' >> ~/.bashrc \
    && echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Install Python 3.7.7
RUN pyenv install 3.7.7 \
    && pyenv global 3.7.7

# Verify Python installation
RUN python --version

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
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# (mini)conda
# https://repo.anaconda.com/miniconda/
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    sh ./Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    export PATH=$PATH:/opt/conda/bin && \
    conda install conda-build

ENV PATH $PATH:/opt/conda/envs/env/bin:/opt/conda/bin

# installing PVnet dependencies (and removing pvnet again)
RUN cd /opt && \
    git clone https://github.com/yangfei4/clean-pvnet.git pvnet && \
    cd pvnet && \
    conda init bash && \
    conda create -n pvnet python=3.7 && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate pvnet && \
    pip install --user torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable && \
    pip install Cython==0.28.2 && \
    pip install -r requirements.txt && \
    pip install --user transforms3d && \
    cd .. && \
    rm -rf pvnet

CMD ["/bin/bash"]
