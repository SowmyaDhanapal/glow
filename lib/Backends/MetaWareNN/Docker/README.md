## Steps to use the docker setup to build and run the Glow
1. To create a docker container with Ubuntu 18.04 as base, run  
        * `sudo bash Docker.sh`  
2. Copy the shell script to docker folder   
        * `scp uname@ip_address:/path/to/local/machine/glow_deps.sh /path/to/docker/folder`  
3. Run the shell script to install the Glow related dependencies  
        * `cd /path/to/docker/folder`  
        * `bash glow_deps.sh`  
        [Note]: The above commands will install all glow related dependencies including, llvm, protobuf, etc., and clones the glow repository. It will take more than an hour to finish the installation.  
4. To build the glow,  
        * `scp uname@ip_address:/path/to/local/machine/lib_protobuf_MWNN_PROTO.zip /path/to/docker/folder/glow/lib/Backends/MetaWareNN`  
        * `cd /path/to/docker/folder/glow/lib/Backends/MetaWareNN`  
        * `unzip /path/to/docker/folder/glow/lib/Backends/MetaWareNN/lib_protobuf_MWNN_PROTO.zip`  
        * `cd /path/to/glow/`  
        * `mkdir build_Release`  
        * `cd build_Release`  
        * `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../../glow -DLLVM_DIR=/path/to/docker/folder/glow/llvm_install/lib/cmake/llvm`  
        * `ninja all`  
### To run the Inference,
Changes in the Glow source code needs the last command (`ninja all`) of step-4, to rebuild the code  
 * #### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]  
   1. Update the "/glow/lib/Backends/MetaWareNN/metawarenn_lib/executable_network/metawarenn_executable_graph.cc" with path to store the MWNNExecutableNetwork.bin in line no: 401 & line no: 414  
   2. Update the "/glow/lib/Backends/MetaWareNN/metawarenn_lib/mwnn_inference_api/mwnn_inference_api.cc" file with saved file path of MWNNExecutableNetwork.bin in line no: 51  
* #### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file  
   1. Update "/glow/lib/Backends/MetaWareNN/MetaWareNNFunction.cpp" file  
        i. Set the path to store the MWNN file dumps in line no: 171  
        ii. Update the path to Glow repository in line no: 180  
   2. Update the "/glow/lib/Backends/MetaWareNN/MetaWareNNFunction.h" file  
      i. Set the INVOKE_NNAC macro to 1 in line no: 16  
   3. Update "/glow/lib/Backends/MetaWareNN/metawarenn_lib/mwnnconvert/mwnn_convert.sh" file  
        i. Set the $EV_CNNMODELS_HOME path in line no: 3  
        ii. Set the absolute path for ARC/cnn_tools/setup.sh file in line no: 4  
        iii. Update the path to Glow with MWNN support in line no: 9 & 22  
        iv. Update the path to evgencnn executable in line no: 10  
        v. Update the Imagenet images path in line no: 20  
        vi. Update `evgencnn` to `evgencnn.pyc` if using the release (not development) version of ARC/cnn_tools in line no: 24  
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will be stored in evgencnn/scripts folder.  

### To Run Inference using MetaWareNN Backend
* Download the MobileNet-V2 model using the zip file from egnyte link - https://multicorewareinc.egnyte.com/dl/2JAUNXlGg0 and unzip the same
* `scp uname@ip_address:/path/to/local/machine/mobilenetv2-7.onnx /path/to/docker/folder/`
* `cd /path/to/glow/build_Release/bin`
* `./image-classifier ../../tests/images/imagenet/cat_285.png -image-mode=0to1 -m=/path/to/docker/mobilenetv2-7.onnx -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN`
