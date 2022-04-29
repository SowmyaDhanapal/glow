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
     * #### To create ONNX Proto from MWNNGraph [Default flow]
        1. By default, `INFERENCE_ENGINE` flag is set to zero in metawarenn_lib/metawarenn_common.h, which will create ONNXProto directly from MWNNGraph and store it in inference/op_onnx_models
        2. Enable `INFERENCE_ENGINE` flag in metawarenn_lib/metawarenn_common.h, to convert MWNNGraph to ExecutableGraph and then create Inference Engine & Execution Context and finally creates the output ONNXProto in inference/op_onnx_models
     * #### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file - Outdated & Optional [Not tested after MWNNGraph update to ONNX format]
        1. Enable the `INVOKE_NNAC` macro in glow/lib/Backends/MetaWareNN/MetaWareNNFunction.h line no: 23
        2. Set the absolute path to ARC/ directory in glow/mwnn_inference/env.sh line no: 11
        3. Set the absolute path to cnn_models/ directory in glow/mwnn_inference/env.sh line no: 12
     ```
     [Note] : Generated EV Binary file for MetaWareNN SubGraph will be stored in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/glow/NNAC_DUMPS` folder
     ```
     * #### To use metawarenn_lib as shared library - Outdated & Optional
        1. Rename lib/Backends/MetaWareNN/CMakeLists.txt to CMakeLists_original.txt
             * `mv lib/Backends/MetaWareNN/CMakeLists.txt lib/Backends/MetaWareNN/CMakeLists_original.txt`
        2. Rename lib/Backends/MetaWareNN/CMakeLists_shared_lib.txt to CMakeLists.txt
             * `mv lib/Backends/MetaWareNN/CMakeLists_shared_lib.txt lib/Backends/MetaWareNN/CMakeLists.txt`
        3. Download the metawarenn shared library from egnyte link https://multicorewareinc.egnyte.com/dl/n31afFTwP9 and place it in `glow/lib/Backends/MetaWareNN/metawarenn_lib/lib`
        4. Also download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `glow/lib/Backends/MetaWareNN/metawarenn_lib/lib`

     * #### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
        1. Set the absolute path to glow in glow/lib/Backends/MetaWareNN/env.sh line no: 5
     * #### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
        1. Enable the `INVOKE_NNAC` macro in glow/lib/Backends/MetaWareNN/MetaWareNNFunction.h line no: 23
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

### To run subgraphs with AvgPool Unsupported node
* `./image-classifier ../../tests/images/imagenet/dog_207.png -image-mode=0to1 -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -cpu-memory=100000 -load-device-configs="../../mwnn_inference/heterogeneousConfigs.yaml"`

## To generate ONNX Proto for multiple Float ONNX models and verify it with original ONNX models
* `cd /path/to/glow/mwnn_inference`
* `source env.sh`
* Download the models from ONNX model zoo by running the below shell script or Download from Egnyte link - https://multicorewareinc.egnyte.com/fl/1hIpgufAHp
(This script will create a folder `onnx_models` inside glow/ directory and download models into it.)
    *   `sh download_ONNX_models.sh`
* Run the ONNX models from model zoo in metawarenn backend with below command
    *   `python test_regression_onnx.py` - Dumps the generated ONNX protos & validation_result.txt(contains output comparison from generated and original ONNX model) in `mwnn_inference/op_onnx_models` directory.
NOTE: Install the following pip packages for verification of ONNX models
* `pip install onnx` current version - 1.10.2
* `pip install onnxruntime` current version - 1.9.0
* `pip install Pillow` current version - 8.4.0

## To generate ONNX Proto for multiple TFLite quantized models and verify it with original TFlite quantized models
* `cd /path/to/glow/mwnn_inference`
* `source env.sh`
* Download the Quantized TFLite models from zoo by running the below shell script or Download Quantized TFLite Models from Egnyte Link - https://multicorewareinc.egnyte.com/fl/7uaWWI9PNi
(This script will create a folder `tflite_quantized_models` inside glow/ directory and download models into it.)
    *   `sh download_quantized_tflite_models.sh`
* Run the Quantized TFLite models from model zoo in metawarenn backend with below command
    *   `python test_regression_quantized_tflite.py` - Dumps the generated ONNX protos & validation_result.txt(contains output comparison from generated and original TFLite models) in `mwnn_inference/op_tflite_quantized_models` directory.
NOTE: Install the following pip packages for verification of ONNX models generated from TFLite models
* `pip install onnx` current version - 1.10.2
* `pip install onnxruntime` current version - 1.9.0
* `pip install Pillow` current version - 8.4.0
* `pip install tflite==2.3.0`
* `pip install decorator` current version - 5.1.0
* `pip install scipy` current version - 1.5.4

### To Run Standalone Inference using MetaWareNN Backend [Deprecated]
* `cd /path/to/glow/mwnn_inference/standalone_inference`
* Download GLOW libraries from this link -> https://multicorewareinc.egnyte.com/dl/ffpW2aAaUm/? and place it in /path/to/glow/mwnn_inference/standalone_inference folder
* `sh run_inference.sh`
* `./inference ../../../../tests/images/imagenet/cat_285.png -m=/path/to/mobilenetv2-7.onnx -model-input-name=data -backend=MetaWareNN`
