#include "clip.h"

namespace metawarenn {

namespace op {

Clip::Clip() { std::cout << "\n In Clip Constructor!!!"; }

Clip::Clip(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs,
         float n_min, float n_max) : Node(n_name, "Clip") {
  inputs = n_inputs;
  outputs = n_outputs;
  min = n_min;
  max = n_max;
  }

void Clip::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Clip fill_attributes!!!";
  std::cout << "\n Min : " << min;
  std::cout << "\n Max : " << max;
  layer_serializer.append(static_cast<float>(min));
  layer_serializer.append(static_cast<float>(max));
  }
} //namespace op

} //namespace metawarenn
