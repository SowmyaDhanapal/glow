# Steps to Build & Run GLOW MetaWareNN Backend

## Use Docker Setup to build and run the Glow
##### Check on the [/glow/lib/Backends/MetaWareNN/Docker/README.md](https://github.com/SowmyaDhanapal/glow/blob/metawarenn_dev/lib/Backends/MetaWareNN/Docker/README.md)

## No Docker Process
### Get Glow
  ### Initial Setup
    1. `git clone --recursive https://github.com/SowmyaDhanapal/glow.git`
    2. `cd glow`
    3. `git checkout metawarenn_dev` (Created metawarenn_dev branch from this master branch commit - 916b8914e0585c220b6186a241db0845c8eff5a9)
    4. `git submodule update --init --recursive`
    5. In case if glow is cloned without metawarenn_lib submodule, use below commands to pull MetaWareNN Library Submodule for the first time
        i.  `git pull`
        ii. `git submodule update --init --recursive`
        iii. Move to metawarenn_lib submodule and checkout to metawarenn_dev branch
            a. `cd lib/Backends/MetaWareNN/metawarenn_lib`
            b. `git checkout metawarenn_dev`

### Using Existing Setup to pull submodule changes [Docker / Non-Docker]
    1. `cd glow`
    2. `git pull`
    3. `cd lib/Backends/MetaWareNN/metawarenn_lib`
    4. `git checkout metawarenn_dev`
    5. `git pull`

### Create a Python Virtual Environment
1. `sudo pip3 install virtualenv`
2. `virtualenv --python=/usr/bin/python3.6 ./venv_glow`
3.  `source ./venv_glow/bin/activate`

### Steps to Build & Install Dependency Packages
* #### Python Packages
    1. `pip install torch torchvision`
    2. `pip install numpy`

* #### To Build FMT
    1. `git clone https://github.com/fmtlib/fmt`
    2. `mkdir fmt/build`
    3. `cd fmt/build`
    4. `cmake ..`
    5. `make`
    6. `sudo make install`

* #### To Build LLVM
    1. `sudo apt-get install libxml2-dev libxml2`
    2. `cd glow`
    3. `source ./utils/build_llvm.sh`

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

* ### Protobuf library dependencies
    * Required Protobuf Version - 3.11.3, Check with the following command,
      $ protoc --version
    * To build protobuf:
        1. `wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-all-3.11.3.tar.gz`
        2. `tar -xf protobuf-all-3.11.3.tar.gz`
        3. `cd protobuf-3.11.3`
        4. `./configure [--prefix=install_protobuf_folder]`
        5. `make`
        6. `make check`
        7. `sudo make install`
        8. `cd ./python`
        9. `python3 setup.py build`
        10. `python3 setup.py test`
        11. `sudo python3 setup.py install`
        12. `sudo ldconfig`
        13. `# if not installed with sudo`
            `export PATH=install_protobuf_folder/bin:${PATH}`
            `export LD_LIBRARY_PATH=install_protobuf_folder/lib:${LD_LIBRARY_PATH}`
            `export CPLUS_INCLUDE_PATH=install_protobuf_folder/include:${CPLUS_INCLUDE_PATH}`

* ### Other necessary dependencies
  ```
  sudo apt-get install graphviz libpng-dev \
    ninja-build wget \
    opencl-headers libgoogle-glog-dev libboost-all-dev \
    libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
    libjemalloc-dev libpthread-stubs0-dev
  ```
* #### Download & store the protobuf file
  ```
   1. Download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `glow/lib/Backends/MetaWareNN/metawarenn_lib/lib`
  ```
* #### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
  ```
   1. Set the absolute path to glow in glow/lib/Backends/MetaWareNN/env.sh line no: 5
  ```
* #### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
  ```
   1. Enable the `INVOKE_NNAC` macro in glow/lib/Backends/MetaWareNN/MetaWareNNFunction.h line no: 19
   2. Set the absolute path to ARC/ directory in glow/lib/Backends/MetaWareNN/env.sh line no: 11
   3. Set the absolute path to cnn_models/ directory in glow/lib/Backends/MetaWareNN/env.sh line no: 12
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will be stored in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/glow/NNAC_DUMPS` folder
  ```
* #### To use metawarenn_lib as shared library
  ```
   1. Rename lib/Backends/MetaWareNN/CMakeLists.txt to CMakeLists_original.txt
      `mv lib/Backends/MetaWareNN/CMakeLists.txt lib/Backends/MetaWareNN/CMakeLists_original.txt`
   2. Rename lib/Backends/MetaWareNN/CMakeLists_shared_lib.txt to CMakeLists.txt
      `mv lib/Backends/MetaWareNN/CMakeLists_shared_lib.txt lib/Backends/MetaWareNN/CMakeLists.txt`
   3. Download the metawarenn shared library from egnyte link https://multicorewareinc.egnyte.com/dl/n31afFTwP9 and place it in `glow/lib/Backends/MetaWareNN/metawarenn_lib/lib`
   4. Also download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `glow/lib/Backends/MetaWareNN/metawarenn_lib/lib`
  ```

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
* Download the MobileNet-V2 model using the zip file from egnyte link - https://multicorewareinc.egnyte.com/dl/2JAUNXlGg0 and unzip the same
* `cd /path/to/glow/build_Release/bin`
* `source /path/to/glow/mwnn_inference/env.sh`
* `./image-classifier ../../tests/images/imagenet/cat_285.png -image-mode=0to1 -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN`
* `./image-classifier ../../tests/images/imagenet/dog_207.png -image-mode=0to1 -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -cpu-memory=100000 -load-device-configs="../tests/runtime_test/heterogeneousConfigs.yaml"`

## To run multiple ONNX models from model zoo
* `cd /path/to/glow/mwnn_inference`
* `source env.sh`
* Download the models from ONNX model zoo by running the below shell script.
(This script will create a folder `onnx_models` inside glow/ directory and download models into it.)
    *   `sh download_ONNX_models.sh`
* Run the ONNX models from model zoo in metawarenn backend with below command
    *   `python inference_regression.py`

## To generate ONNX Proto and verify it with original ONNX models
* `cd /path/to/glow/mwnn_inference`
* `source env.sh`
* Download the models from ONNX model zoo by running the below shell script.
(This script will create a folder `onnx_models` inside glow/ directory and download models into it.)
    *   `sh download_ONNX_models.sh`
* Run the ONNX models from model zoo in metawarenn backend with below command
    *   `python validate_models.py`
NOTE: Install the following pip packages for verification of ONNX models
* `pip install onnx` current version - 1.10.2
* `pip install onnxruntime` current version - 1.9.0
* `pip install Pillow` current version - 8.4.0

### To Run Standalone Inference using MetaWareNN Backend (deprecated)
* `cd /path/to/glow/lib/Backends/MetaWareNN/Inference`
* Download GLOW libraries from this link -> https://multicorewareinc.egnyte.com/dl/ffpW2aAaUm/? and place it in /path/to/glow/lib/Backends/MetaWareNN/Inference folder
* `sh run_inference.sh`
* `./inference ../../../../tests/images/imagenet/cat_285.png -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -backend=MetaWareNN`
