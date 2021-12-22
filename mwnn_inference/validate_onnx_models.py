import os
import shutil
import onnx
import numpy as np
from PIL import Image
import onnxruntime
import subprocess

glow_path = os.environ['FRAMEWORK_PATH']
model_dir = glow_path + "/onnx_models/"
f = open(glow_path + "/mwnn_inference/models.txt", "r")

output_folder = glow_path + '/mwnn_inference/op_onnx_models/'

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
  elif model_name == "tinyyolov2-7.onnx" or model_name == "yolov2-coco-9.onnx":
    image_path = glow_path + "/mwnn_inference/image/yolov2_416.png"
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

  elif(model_name == "tinyyolov2-7.onnx" or model_name == "yolov2-coco-9.onnx"):
    subprocess.run([glow_path+ "/build_Release/bin/object-detector", image_path, "-m", model_path, "-model-input-name", input_node.name, "-cpu-memory", "100000", "-backend=MetaWareNN", "-onnx-define-symbol=None,1"])
  else:
    subprocess.run([glow_path+ "/build_Release/bin/image-classifier", image_path, "-m", model_path, "-model-input-name", input_node.name, "-cpu-memory", "100000", "-backend=MetaWareNN"])

  gen_model_name = glow_path + "/mwnn_inference/op_onnx_models/model_" + model_name
  os.rename(glow_path + "/mwnn_inference/model.onnx", gen_model_name)

  session = onnxruntime.InferenceSession(model_path, None)
  input_dict = {}
  for input in session.get_inputs():
      print('input1: ', input.name)
      print('input shape: ', input.shape, type(input.shape))
      shape = []
      for dim in input.shape:
          if isinstance(dim, str):
              shape.append(1)
          else:
              shape.append(dim)
      data = np.random.random_sample(shape)
      if input.type == 'tensor(float)':
          data = data.astype(np.float32)
      else:
          print('not valid input datatype!')
          exit()
      input_dict[input.name] = data
  print(input_dict[input.name].shape)
  input_name = session.get_inputs()[0].name
  output_name = []
  for out in session.get_outputs():
      output_name.append(out.name)
  print('input_name :', input_name)
  print('output_name :', output_name)
  result  = session.run(output_name, {input_name: input_dict[input_name]})
  result_arr = np.array(result)
  flat_result = result_arr.flatten()
  flat_result[::-1].sort()

  if(model_name == "efficientnet-lite4-11.onnx"):
    new_data = np.rollaxis(data, 3, 1)
  else:
    new_data = input_dict[input_name]
  session_mwnn = onnxruntime.InferenceSession(gen_model_name, None)  #Updated model name
  input_name_mwnn = session_mwnn.get_inputs()[0].name
  output_name_mwnn = []
  for out in session_mwnn.get_outputs():
      output_name_mwnn.append(out.name)
  print('input_name_mwnn :', input_name_mwnn)
  print('output_name_mwnn :', output_name_mwnn)
  result_mwnn  = session_mwnn.run(output_name_mwnn, {input_name_mwnn: new_data}) #Layer dump name same as above
  result_arr_mwnn = np.array(result_mwnn)
  flat_result_mwnn = result_arr_mwnn.flatten()
  flat_result_mwnn[::-1].sort()
  op_file.write("\n=====================================================================================================\n")
  op_file.write(model_path)
  for idx in range(0, 5):
    op_file.write("\nDefault : " + str(flat_result[idx]) + "      New : " + str(flat_result_mwnn[idx]))
    error = abs(flat_result[idx]-flat_result_mwnn[idx])
    if(error > 1e-05):
      op_file.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MISMATCH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
op_file.close()
