#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class BatchNormalization : public Node {
  public:
    BatchNormalization();
    BatchNormalization(std::string n_name, std::vector<std::string> n_inputs,
            std::vector<std::string> n_outputs, float n_epsilon, float n_momentum);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    float epsilon;
    float momentum;
};

} //namespace op

} //namespace metawarenn
