FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PATH="/root/anaconda3/bin:$PATH"
# \TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

RUN apt-get update --allow-releaseinfo-change && apt-get install -y libgbm-dev -y \
    software-properties-common dirmngr -y \
    build-essential -y \
    libgl1-mesa-glx libxrender1 libfontconfig1 -y \
    libglib2.0-0 -y \
    libsm6 libxext6 libxrender-dev -y \
    vim zip unzip wget screen -y \
    gnupg2 -y \
    libgl1-mesa-glx -y \
    git libmagickwand-dev -y

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        python3-dev \
        python3-opencv

# RUN cd /root && wget https://repo.anaconda.com/archive/Anaconda3-2023.07-0-Linux-x86_64.sh \
#     && cd /root && bash Anaconda3-2023.07-0-Linux-x86_64.sh -b -p ./anaconda3 \
#     && bash -c "source /root/anaconda3/etc/profile.d/conda.sh"

# ENV PATH /opt/conda/bin:$PATH

# Install Packages
COPY requirements.txt /
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip3 install -r requirements.txt

# Install tensorflow
# RUN pip3 install nvidia-cudnn-cu11==8.6.0.163
RUN pip3 install tensorflow==2.12.0


# Copy all files in the current directory to the working directory
WORKDIR /app
COPY . /app


CMD [ "/bin/bash" ]