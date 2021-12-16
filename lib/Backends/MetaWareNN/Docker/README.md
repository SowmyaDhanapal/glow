## Steps to use the docker setup to build and run the Glow
1. To create a docker container with Ubuntu 18.04 as base, run
     * `sudo bash Docker.sh`
2. Copy the shell script to docker folder
     * `cp /path/to/local/machine/glow_deps.sh /path/to/docker/folder/root`
3. Run the shell script to install the Glow related dependencies
     * `cd /path/to/docker/folder/root`
     * `bash glow_deps.sh`
        [Note]: The above commands will install all glow related dependencies including llvm, clang, cmake, protobuf, fmt etc., and clones the glow repository. It will take about an hour to finish the installation.
4. Update necessary paths
     * #### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
        1. Set the absolute path to glow in glow/lib/Backends/MetaWareNN/env.sh line no: 5
     * #### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
        1. Enable the `INVOKE_NNAC` macro in glow/lib/Backends/MetaWareNN/MetaWareNNFunction.h line no: 19
        2. Set the absolute path to ARC/ directory in glow/lib/Backends/MetaWareNN/env.sh line no: 11
        3. Set the absolute path to cnn_models/ directory in glow/lib/Backends/MetaWareNN/env.sh line no: 12
              * [Note] : Generated EV Binary file for MetaWareNN SubGraph will be stored in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/glow/NNAC_DUMPS` folder
5. Set the Environmental Variables for Build & Inference
     * `source /path/to/docker/glow/lib/Backends/MetaWareNN/env.sh`
6. To build the glow
     * `cd /path/to/glow/`
     * `mkdir build_Release`
     * `cd build_Release`
     * `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../../glow -DLLVM_DIR=$FRAMEWORK_PATH/llvm_install/lib/cmake/llvm`
     * `ninja all`

## To Run Inference using MetaWareNN Backend
* Download the MobileNet-V2 model using the zip file from egnyte link - https://multicorewareinc.egnyte.com/dl/2JAUNXlGg0 and unzip the same
* `scp uname@ip_address:/path/to/local/machine/mobilenetv2-7.onnx /path/to/docker/folder/`
* `cd $FRAMEWORK_PATH/build_Release/bin`
* `./image-classifier ../../tests/images/imagenet/cat_285.png -image-mode=0to1 -m=/path/to/docker/mobilenetv2-7.onnx -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN`

## To Run Multiple ONNX models from model zoo
* `cd $FRAMEWORK_PATH/build_Release/bin`
* Download the models from ONNX model zoo by running the below shell script.
(This script will create a folder `onnx_models` inside glow/ directory and download models into it.)
    *   `sh $FRAMEWORK_PATH/lib/Backends/MetaWareNN/download_ONNX_models.sh`
* Run the ONNX models from model zoo in metawarenn backend with below command
    *   `sh $FRAMEWORK_PATH/lib/Backends/MetaWareNN/run_ONNX_models.sh`

## To generate ONNX Proto and verify it with original ONNX models
* `cd /path/to/glow/mwnn_inference`
* `source env.sh`
* Download the models from ONNX model zoo by running the below shell script.
(This script will create a folder `onnx_models` inside glow/ directory and download models into it.)
    *   `sh download_ONNX_models.sh`
* Run the ONNX models from model zoo in metawarenn backend with below command
    *   `python validate_models.py`
