#include "batch_normalization.h"

namespace metawarenn {

namespace op {

BatchNormalization::BatchNormalization() { std::cout << "\n In BatchNormalization Constructor!!!"; }

BatchNormalization::BatchNormalization(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs, float n_epsilon, float n_momentum) : Node(n_name, "BatchNormalization") {
  inputs = n_inputs;
  outputs = n_outputs;
  epsilon = n_epsilon;
  momentum = n_momentum;
  }

void BatchNormalization::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In BatchNormalization fill_attributes!!!";
  std::cout << "\n Epsilon : " << epsilon;
  layer_serializer.append(static_cast<float>(epsilon));
  std::cout << "\n Momentum : " << momentum;
  layer_serializer.append(static_cast<float>(momentum));
  }
} //namespace op

} //namespace metawarenn
