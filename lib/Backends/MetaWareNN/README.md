## Steps to Build & Run GLOW MetaWareNN Backend
### Get Glow 
1. `git clone https://github.com/SowmyaDhanapal/glow.git`
2. `cd glow`
3. `git submodule update --init --recursive`
4. `git checkout metawarenn_dev` (Created metawarenn_dev branch from this master branch commit - 916b8914e0585c220b6186a241db0845c8eff5a9)

### Create a Python Virtual Environment
1. `sudo pip3 install virtualenv`
2. `virtualenv --python=/usr/bin/python3.6 ./venv_glow`
3.  `source ./venv_glow/bin/activate`

### Steps to Build & Install Dependency Packages
* #### Python Packages
    1. `pip install torch`
    2. `pip install numpy`
    3. `pip install torchvision`

* #### To Build FMT
    1. `git clone https://github.com/fmtlib/fmt`
    2. `mkdir fmt/build`
    3. `cd fmt/build`
    4. `cmake ..`
    5. `make`
    6. `sudo make install`

* #### To Build LLVM
    1. `cd ../../glow`
    2. `./utils/build_llvm.sh`

* #### To Build Required CMake Version
    [Note]: Glow Installation is successful with CMake of version 3.16.5, So check the CMake Version using the below command,
    * `cmake --version`

    If CMake Version is not matched, install the required version using the following steps,
    * `wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz`
    * `tar -zxvf cmake-3.16.5.tar.gz`
    * `cd cmake-3.16.5`
    * `./bootstrap`
    * `make`
    * `sudo make install`
    
### Configure and Build Glow
* #### For Release Build 
    * `mkdir build_Release`
    * `cd build_Release`
    * `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../../glow -DLLVM_DIR=/path/to/glow/llvm_install/lib/cmake/llvm`
    * `ninja all`
* #### For Debug Build
    *  `mkdir build_Debug`
    *  `cd build_Debug`
    *  `cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../../glow -DLLVM_DIR=/path/to/glow/llvm_install/lib/cmake/llvm`
    *  `ninja all`

### To Run Inference using MetaWareNN Backend
* `Download the model at https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx`
* `cd /path/to/glow/build_Release/bin`
* `./image-classifier ../../tests/images/imagenet/cat_285.png -image-mode=0to1 -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN`
