#include "channel_shuffle.h"

namespace metawarenn {

namespace op {

ChannelShuffle::ChannelShuffle(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "ChannelShuffle") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
