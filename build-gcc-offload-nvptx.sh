#!/bin/bash

#
# Build GCC with support for offloading to NVIDIA GPUs.
#

set -o nounset -o errexit

# Location of the installed CUDA toolkit
cuda=/usr/local/cuda-8.0


# directory of this script
MYDIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

work_dir=$MYDIR/gcc-offload
install_dir=$work_dir/install

rm -rf $work_dir

# Build assembler and linking tools
mkdir -p $work_dir
cd $work_dir
git clone https://github.com/MentorEmbedded/nvptx-tools
cd nvptx-tools
./configure \
	--with-cuda-driver-include=$cuda/include \
	--with-cuda-driver-lib=$cuda/lib64 \
	--prefix=$install_dir
make
make install
cd ..


# Set up the GCC source tree
git clone https://github.com/MentorEmbedded/nvptx-newlib
wget -c http://gnu.mirror.globo.tech/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz
tar xf gcc-7.3.0.tar.gz
cd gcc-7.3.0
contrib/download_prerequisites
ln -s ../nvptx-newlib/newlib newlib
target=$(./config.guess)
cd ..


# Build nvptx GCC
mkdir build-nvptx-gcc
cd build-nvptx-gcc
../gcc-7.3.0/configure \
	--target=nvptx-none \
	--with-build-time-tools=$install_dir/nvptx-none/bin \
	--enable-as-accelerator-for=$target \
	--disable-sjlj-exceptions \
	--enable-newlib-io-long-long \
	--enable-languages="c,c++,fortran,lto" \
	--prefix=$install_dir
make -j4
make install
cd ..


# Build host GCC
mkdir build-host-gcc
cd  build-host-gcc
../gcc-7.3.0/configure \
	--enable-offload-targets=nvptx-none \
	--with-cuda-driver-include=$cuda/include \
	--with-cuda-driver-lib=$cuda/lib64 \
	--disable-bootstrap \
	--disable-multilib \
	--enable-languages="c,c++,fortran,lto" \
	--prefix=$install_dir
make -j4
make install
cd ..
