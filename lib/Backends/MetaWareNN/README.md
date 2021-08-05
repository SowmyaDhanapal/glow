## Steps to Build & Run GLOW MetaWareNN Backend
### Get Glow
1. `git clone https://github.com/SowmyaDhanapal/glow.git`
2. `cd glow`
3. `git checkout metawarenn_dev` (Created metawarenn_dev branch from this master branch commit - 916b8914e0585c220b6186a241db0845c8eff5a9)
4. `git submodule update --init --recursive`  
  
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
    1. `cd glow`
    2. `source ./utils/build_llvm.sh`

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
           
    * Download protobuf library version 3.11.3 from the egnyte link https://multicorewareinc.egnyte.com/dl/FjljPlgjlI
    * Unzip and move the "libprotobuf.so" to "/path/to/glow/lib/Backends/MetaWareNN"

* #### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
   1. Update the "/glow/lib/Backends/MetaWareNN/metawarenn_lib/executable_network/metawarenn_executable_graph.cc" with path to store the MWNNExecutableNetwork.bin in line no: 401 & line no: 414
   2. Update the "/glow/lib/Backends/MetaWareNN/metawarenn_lib/mwnn_inference_api/mwnn_inference_api.cc" file with saved file path of MWNNExecutableNetwork.bin in line no: 51
* #### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
   1. Update "/glow/lib/Backends/MetaWareNN/MetaWareNNFunction.cpp" file
        i. Set the path to store the MWNN file dumps in line no: 171
        ii. Update the path to Glow repository in line no: 180
   2. Update the "/glow/lib/Backends/MetaWareNN/MetaWareNNFunction.h" file as follows:
      i. Set the INVOKE_NNAC macro to 1 in line no: 16
   3. Update "/glow/lib/Backends/MetaWareNN/metawarenn_lib/mwnnconvert/mwnn_convert.sh" file
        i. Set the $EV_CNNMODELS_HOME path in line no: 3
        ii. Set the absolute path for ARC/cnn_tools/setup.sh file in line no: 4
        iii. Update the path to Glow with MWNN support in line no: 9 & 20
        iv. Update the path to evgencnn executable in line no: 10
        v. Update the Imagenet images path in line no: 18
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will store in evgencnn/scripts folder.
* ### Modify the below mentioned files
    1. Update "/glow/lib/Backends/MetaWareNN/MetaWareNNFunction.cpp" file
        i. Set the path to store the MWNN file dumps in line no: 171
        ii. Update the path to Glow repository in line no: 180
    2. Update "/glow/lib/Backends/MetaWareNN/metawarenn_lib/mwnnconvert/mwnn_convert.sh" file
        i. Set the $EV_CNNMODELS_HOME path in line no: 3
        ii. Set the absolute path for ARC/cnn_tools/setup.sh file in line no: 4
        iii. Update the path to Glow with MWNN support in line no: 9 & 20
        iv. Update the path to evgencnn executable in line no: 10
        v. Update the Imagenet images path in line no: 18
    3. Update the "/glow/lib/Backends/MetaWareNN/metawarenn_lib/mwnn_inference_api/mwnn_inference_api.cc" file
        i.  Set the path to evgencnn/scripts folder in line no: 51
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
* `./image-classifier ../../tests/images/imagenet/cat_285.png -image-mode=0to1 -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN`
* `./image-classifier ../../tests/images/imagenet/dog_207.png -image-mode=0to1 -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -cpu-memory=100000 -load-device-configs="../tests/runtime_test/heterogeneousConfigs.yaml"`

## To run multiple ONNX models from model zoo
* Create a directory to download onnx models and move to the directory
* `cd /path/to/store/onnx_model_dir`
* Download the models from ONNX model zoo by running the below shell script
    *   `sh /path/to/glow/lib/Backends/MetaWareNN/download_ONNX_models.sh`
* Set the path to downloaded onnx model directory in glow/lib/Backends/MetaWareNN/run_ONNX_models.sh file line no: 1
* `cd /path/to/glow/build_Release/bin`
* Run the ONNX models from model zoo in metawarenn backend with below command
    *   `sh /path/to/glow/lib/Backends/MetaWareNN/run_ONNX_models.sh`

* Note: MobileNet-V2 model is only passed through execution flow, since MLI kernels are yet to be added for additional operators in the models from model zoo. So, recently updated MobileNet-V2 model(with additional operators) from model zoo is excluded in this shell script.

### To Run Standalone Inference using MetaWareNN Backend
* `cd /path/to/glow/lib/Backends/MetaWareNN/Inference`
* Download GLOW libraries from this link -> https://multicorewareinc.egnyte.com/dl/ffpW2aAaUm/? and place it in /path/to/glow/lib/Backends/MetaWareNN/Inference folder
* `sh run_inference.sh`
* `./inference ../../../../tests/images/imagenet/cat_285.png -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -backend=MetaWareNN`
