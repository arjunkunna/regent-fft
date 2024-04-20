#!/bin/bash

#test1
export INCLUDE_PATH="$INCLUDE_PATH;$PWD/fftw-3.3.8/install/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/fftw-3.3.8/install/lib"
export TERRA_PATH="$TERRA_PATH;$PWD/src/?.rg"

set -ex

#sudo apt-get update -qq
#sudo apt-get install -qq software-properties-common

#wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
#sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
#sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
#sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

#sudo apt-get install nvidia-driver-535
#sudo apt-get update -qq
#sudo apt-get install -qq cuda-toolkit-12-2
#sudo apt-get install nvidia-utils-515
#sudo apt install -y libjemalloc2 # valgrind

#LIBJEMALLOC_SO="libjemalloc.so.2"
#echo "Checking libjemalloc.so.2"
#LIBJEMALLOC_SO_PATH="$(whereis -b "$LIBJEMALLOC_SO" | cut -d ' ' -f2 | tr -d '\n')"
#if [[ -n $LIBJEMALLOC_SO_PATH ]]; then
#    echo "Found: $LIBJEMALLOC_SO_PATH"
#else
#    echo "Error: $LIBJEMALLOC_SO not found"
#    exit 1
#fi

export CUDA_PATH="/usr/local/cuda"
export CUDA="/usr/local/cuda"

export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

which nvcc
#ls -l /usr/local
#ls -l /usr/local/cuda
#ls -lH /usr/local/cuda
#ls -lH /usr/local/cuda/include
#ls -lH /usr/local/cuda/lib64
#ls -lH /usr/lib64
#ls -lH /usr/lib
#ls -lH /usr/lib/x86_64-linux-gnu

#find /usr/ -name 'libcuda.so.*'
#nvidia-smi

echo "CUDA_PATH=${CUDA_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

nvcc --version
git clone --depth 1 --branch stable https://github.com/StanfordLegion/legion.git
CC=gcc CXX=g++ DEBUG=${DEBUG:-0} USE_GASNET=0 USE_CUDA=1 ./legion/language/scripts/setup_env.py --terra-binary
./install.py
#LD_PRELOAD="$LIBJEMALLOC_SO_PATH" ./legion/language/regent.py test/fft_test.rg -fgpu cuda -fgpu-offline 1 -fgpu-arch pascal
./legion/language/regent.py test/fft_test.rg -fgpu cuda -fgpu-offline 1 -fgpu-arch pascal

