import tensorflow as tf
import tvm
from tvm import relay
import tflite
import numpy as np
import os
import shutil
from PIL import Image
import onnx
import onnxruntime
import subprocess

FLOAT_OUTPUT = False
if(FLOAT_OUTPUT):
  threshold = 0.1
else:
  threshold = 10

glow_path = os.environ['FRAMEWORK_PATH']
op_dump_folder = glow_path + '/mwnn_inference/op_tflite_quantized_models'
if os.path.exists(op_dump_folder):
    shutil.rmtree(op_dump_folder)
os.makedirs(op_dump_folder)
op_file = open(op_dump_folder + "/validation_result.txt", 'w')
image_path = glow_path + "/tests/images/imagenet/cat_285.png"
dtype = "float32"
model_dir = glow_path + "/tflite_quantized_models/"
f = open("tflite_quantized_models.txt", "r")
img = Image.open("image/kitten.jpg")

for line in f:
  inp_shape = []
  model_name = line.strip()
  head, tail = os.path.split(model_name)
  os.environ["MODELNAME"] = tail
  model_path = model_dir + model_name
  print("Model path: ", model_path)
  if(os.path.exists(model_path)):
    tflite_model_buf = open(model_path, "rb").read()
    # Get TFLite model from buffer
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
  else:
    print("Please check the model path")
    exit(1)

#======================================================================================================
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test the model on random input data.
  input_shape = input_details[0]['shape']
  print("input_shape : ", input_shape, len(input_shape))
  tf_scale = output_details[0]['quantization_parameters']['scales'][0]
  tf_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
  if(input_shape[1] == 299):
    image_path = glow_path + "/mwnn_inference/image/image_299.png"
  img = img.resize((input_shape[1], input_shape[2])) #H, W from model

  # add N dim
  input_data = np.expand_dims(img, axis=0)
  print(np.shape(input_data))
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data)

  result_arr = np.array(output_data)
  flat_result = result_arr.flatten()
  flat_result[::-1].sort()
  if(FLOAT_OUTPUT):
    flat_result = (flat_result - tf_zero_point) * tf_scale
  print(input_details[0]['name'])
  subprocess.run([glow_path+ "/build_Release/bin/image-classifier", image_path, "-m", model_path, "-model-input-name", input_details[0]['name'], "-cpu-memory", "100000", "-backend=MetaWareNN", "-tflite-uint8-to-int8=false", "-tflite-float-softmax=false", "-image-layout", "NHWC"])#, "quantization-schema=symmetric with uint8"])
  gen_model_name = op_dump_folder + "/model_" + str(model_name.split("/")[-1]).split(".tflite")[0] + ".onnx"
  os.rename("model.onnx", gen_model_name)

  #=======================================================================================================
  quant_model_info = onnx.load(gen_model_name)
  graph_def = quant_model_info.graph
  initializers = graph_def.initializer
  nodes = graph_def.node
  outputs = graph_def.output
  quant_op_name = outputs[0].name

  for node in nodes:
    if(node.op_type == "QuantizeLinear" and node.output[0] == quant_op_name):
      for initializer in initializers:
        if(node.input[1] == initializer.name):
          onnx_scale = initializer.float_data[0]
        elif(node.input[2] == initializer.name):
          onnx_zero_point = initializer.int32_data[0]
  
  session_mwnn = onnxruntime.InferenceSession(gen_model_name, None)  #Updated model name
  input_name_mwnn = session_mwnn.get_inputs()[0].name
  output_name_mwnn = []
  for out in session_mwnn.get_outputs():
      output_name_mwnn.append(out.name)
  print('input_name_mwnn :', input_name_mwnn)
  print('output_name_mwnn :', output_name_mwnn)
  new_input =  input_data.astype(np.uint8)
  new_data_onnx = np.rollaxis(new_input, 3, 1)
  result_mwnn = session_mwnn.run(output_name_mwnn, {input_name_mwnn: new_data_onnx}) #layer name

  result_arr_mwnn = np.array(result_mwnn)
  flat_result_mwnn = result_arr_mwnn.flatten()
  flat_result_mwnn[::-1].sort()
  if(FLOAT_OUTPUT):
    flat_result_mwnn = onnx_scale * (flat_result_mwnn-onnx_zero_point)
  op_file.write("\n=====================================================================================================\n")
  op_file.write(model_path)
  for idx in range(0, 5):
    op_file.write("\nDefault : " + str(flat_result[idx]) + "      New : " + str(flat_result_mwnn[idx]))
    error = abs((float)(flat_result[idx])-(float)(flat_result_mwnn[idx]))
    if(error > threshold):
      op_file.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MISMATCH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
op_file.close()