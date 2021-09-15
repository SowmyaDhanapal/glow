mkdir GLOW
cd GLOW
apt -y update
apt -y install git
apt-get -y install wget
apt-get -y install unzip
apt-get -y install openssh-client
apt-get -y install gedit vim
apt-get -y install libssl-dev
apt-get -y install graphviz libpng-dev ninja-build wget opencl-headers libgoogle-glog-dev libboost-all-dev libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev libjemalloc-dev libpthread-stubs0-dev
apt-get -y install python3-pip
python3 -m pip install --upgrade pip
pip3 install torch torchvision
apt-get -y install libxml2-dev libxml2

wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar -zxvf cmake-3.16.5.tar.gz
cd cmake-3.16.5
./configure
make
make install
cd ..
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-all-3.11.3.tar.gz
tar -xf protobuf-all-3.11.3.tar.gz
cd protobuf-3.11.3
./configure
make
make check
make install
cd ./python
python3 setup.py build
python3 setup.py test
python3 setup.py install
export PATH=/usr/local/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/include:${CPLUS_INCLUDE_PATH}
cd ../..

git clone https://github.com/fmtlib/fmt
mkdir fmt/build
cd fmt/build
cmake ..
make
make install
cd ../..

git clone --recursive https://github.com/SowmyaDhanapal/glow.git
cd glow
git checkout metawarenn_dev
git submodule update --init --recursive
cd lib/Backends/MetaWareNN/metawarenn_lib
git checkout metawarenn_dev
cd ../../../..
source ./utils/build_llvm.sh
apt install -y clang-6.0
ln -s /usr/bin/clang-6.0 /usr/bin/clang
ln -s /usr/bin/clang++-6.0 /usr/bin/clang++
