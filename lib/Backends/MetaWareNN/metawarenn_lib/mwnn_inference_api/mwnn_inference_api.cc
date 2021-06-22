#include "mwnn_inference_api.h"

namespace metawarenn {

int MWNNInferenceApi::graph_count = 0;
unsigned long int MWNNInferenceApi::available_bytes = TOTAL_MEMORY_SIZE;
unsigned long int MWNNInferenceApi::used_bytes = 0;

MWNNInferenceApi::MWNNInferenceApi() {
  std::cout << "\n In MWNNInferenceApi";
  if(graph_count == 0)
      mwnn_shm = MWNNSharedMemory();
  graph_count++;
}

void MWNNInferenceApi::prepareInput(float* ip_tensor, std::vector<int> shape) {
  std::cout << "\n In prepareInput";
  unsigned long int total_size = 1;
  for(auto item: shape){
    std::cout << "\n val: " << item;
    total_size = total_size * item;
  }
  unsigned long int ip_size = total_size*(sizeof(float));
  if(this->available_bytes > ip_size) {
      this->input = mwnn_shm.shmp + this->used_bytes;
      memcpy(this->input, ip_tensor, ip_size);
      this->used_bytes = this->used_bytes + ip_size;
      this->available_bytes = this->available_bytes - ip_size;
  }
}

void MWNNInferenceApi::prepareOutput(std::vector<int> shape) {
  std::cout << "\n In prepareOutput";
  unsigned long int total_size = 1;
  for(auto item: shape){
    std::cout << "\n val: " << item;
    total_size = total_size * item;
  }

  unsigned long int op_size = total_size*(sizeof(float));
  if(this->available_bytes > op_size) {
      this->output = mwnn_shm.shmp + this->used_bytes;
      this->used_bytes = this->used_bytes + op_size;
      this->available_bytes = this->available_bytes - op_size;
  }
}

void MWNNInferenceApi::prepareGraph(std::string name) {
  std::cout << "\n In prepareGraph";
  std::ifstream in;
  in.open("/path/to/ARC/cnn_tools/utils/tools/evgencnn/scripts/cnn_bin_" + name + ".bin", std::ios::in | std::ios::binary);
  if(in.is_open()) {
    std::streampos start = in.tellg();
    in.seekg(0, std::ios::end);
    std::streampos end = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> contents;
    contents.resize(static_cast<size_t>(end - start));
    in.read(&contents[0], contents.size());
    auto data = contents.data();
    unsigned long int model_size = contents.size();
    if(this->available_bytes > model_size) {
      this->model = mwnn_shm.shmp + this->used_bytes;
      memcpy(this->model, data, model_size);
      this->used_bytes = this->used_bytes + model_size;
      this->available_bytes = this->available_bytes - model_size;
    }
  }
  else {
    std::cout << "\n Couldn't open the binary file in MWNNInference API!!!!!";
    exit(1);
  }
}

void MWNNInferenceApi::runGraph() {
  std::cout << "\n In runGraph";
  //assume run() call takes input & model binary from shared memory & writes op to shared memory
}

void MWNNInferenceApi::getOutput(float* op_tensor, std::vector<int> shape) {
  std::cout << "\n In getOutput";
  unsigned long int total_size = 1;
  for(auto item: shape) {
    std::cout << "\n val: " << item;
    total_size = total_size * item;
  }
  unsigned long int op_size = total_size * (sizeof(float));
  memcpy(op_tensor, this->output, op_size);
}
} //metawarenn
