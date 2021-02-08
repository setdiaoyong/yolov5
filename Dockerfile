<<<<<<< HEAD
FROM nvidia/cuda:9.2-devel-ubuntu16.04
LABEL maintainer="nweir <nweir@iqt.org>"
=======
# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.12-py3

# Install linux packages
RUN apt update && apt install -y screen libgl1-mesa-glx

# Install python dependencies
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt gsutil
>>>>>>> pr/1

ENV CUDNN_VERSION 7.3.0.29
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
ARG solaris_branch='master'


<<<<<<< HEAD
# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
		libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
		vim \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
=======
# Copy weights
#RUN python3 -c "from models import *; \
#attempt_download('weights/yolov5s.pt'); \
#attempt_download('weights/yolov5m.pt'); \
#attempt_download('weights/yolov5l.pt')"
>>>>>>> pr/1

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

<<<<<<< HEAD
# prepend pytorch and conda-forge before default channel
RUN conda update conda && \
    conda config --prepend channels conda-forge && \
    conda config --prepend channels pytorch

# get dev version of solaris and create conda environment based on its env file
WORKDIR /tmp/
RUN git clone https://github.com/cosmiq/solaris.git && \
    cd solaris && \
    git checkout ${solaris_branch} && \
    conda env create -f environment-gpu.yml
ENV PATH /opt/conda/envs/solaris/bin:$PATH

RUN cd solaris && pip install .

# install various conda dependencies into the space_base environment
RUN conda install -n solaris \
                     jupyter \
                     jupyterlab \
                     ipykernel

# add a jupyter kernel for the conda environment in case it's wanted
RUN source activate solaris && python -m ipykernel.kernelspec \
    --name solaris --display-name solaris

RUN source activate solaris && conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

RUN source activate solaris && pip install tensorboard --upgrade

# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006

RUN ["/bin/bash"]

=======
# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t
# for v in {300..303}; do t=ultralytics/coco:v$v && sudo docker build -t $t . && sudo docker push $t; done

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# Send weights to GCP
# python -c "from utils.general import *; strip_optimizer('runs/train/exp0_*/weights/best.pt', 'tmp.pt')" && gsutil cp tmp.pt gs://*.pt

# Clean up
# docker system prune -a --volumes
>>>>>>> pr/1
