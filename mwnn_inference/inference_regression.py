import os
import shutil
import onnx
from PIL import Image
import subprocess

glow_path = os.environ['FRAMEWORK_PATH']
model_dir = glow_path + "/onnx_models/"
f = open(glow_path + "mwnn_inference/models.txt", "r")

output_folder = glow_path + '/mwnn_inference/conversion_output/'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)
op_file = open(output_folder + "/validation_result.txt", 'w')

for line in f:
  inp_shape = []
  model_name = line.strip()
  model_path = model_dir + model_name
  print("Model path: ", model_path)
  if(os.path.exists(model_path)):
    onnx_model = onnx.load(model_dir + model_name)
  else:
    print("Please check the model path")
    exit(1)
  if model_name == "mnist-7.onnx":
    image_path = glow_path + "/tests/images/mnist/0_1009.png"
  else:
    image_path = glow_path + "/tests/images/imagenet/cat_285.png"

  input_all = [node.name for node in onnx_model.graph.input]
  input_initializer =  [node.name for node in onnx_model.graph.initializer]
  net_feed_input = list(set(input_all)  - set(input_initializer))
  for node in onnx_model.graph.input:
    if node.name == net_feed_input[0]:
      input_node = node
      break
  if(model_name == "mobilenetv2-7.onnx"):
    subprocess.run([glow_path+ "/build_Release/bin/image-classifier", image_path, "-m", model_path, "-model-input-name", input_node.name, "-cpu-memory", "100000", "-backend=MetaWareNN", "-onnx-define-symbol=batch_size,1"])

  elif(model_name == "efficientnet-lite4-11.onnx"):
    subprocess.run([glow_path+ "/build_Release/bin/image-classifier", image_path, "-m", model_path, "-model-input-name", input_node.name, "-image-layout", "NHWC", "-cpu-memory", "100000", "-backend=MetaWareNN"])

  else:
    subprocess.run([glow_path+ "/build_Release/bin/image-classifier", image_path, "-m", model_path, "-model-input-name", input_node.name, "-cpu-memory", "100000", "-backend=MetaWareNN"])