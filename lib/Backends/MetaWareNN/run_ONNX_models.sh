model_path="/path/to/onnx_model_dir/"
image_path="../../tests/images/imagenet/cat_285.png"
./image-classifier $image_path -image-mode=0to1 -m=$model_path"bvlcalexnet-7.onnx" -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"googlenet-7.onnx" -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"inception-v1-7.onnx" -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"inception-v2-7.onnx" -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"rcnn-ilsvrc13-7.onnx" -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet152-v1-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet152-v2-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet18-v1-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet18-v2-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet34-v1-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet34-v2-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet101-v1-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet101-v2-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet50-v1-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet50-v2-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"resnet50-caffe2-v1-7.onnx"-model-input-name=gpu_0/data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"caffenet-7.onnx" -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"densenet-7.onnx" -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"squeezenet1.0-7.onnx" -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"shufflenet-7.onnx" -model-input-name=gpu_0/data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"vgg16-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"vgg19-7.onnx" -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"vgg16-bn-7.onnx"  -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"vgg19-bn-7.onnx"  -model-input-name=data -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"vgg19-caffe2-7.onnx"  -model-input-name=data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"zfnet512-7.onnx"  -model-input-name=gpu_0/data_0 -cpu-memory=100000 -backend=MetaWareNN
./image-classifier $image_path -image-mode=0to1 -m=$model_path"efficientnet-lite4-11.onnx"  -model-input-name=images:0 -image-layout=NHWC -label-offset=1 -cpu-memory=100000 -backend=MetaWareNN
#./image-classifier $image_path -image-mode=0to1 -m=$model_path"mobilenetv2-7.onnx"  -model-input-name=input -cpu-memory=100000 -backend=MetaWareNN -onnx-define-symbol=batch_size,1