#include "MetaWareNN.h"

namespace metawarenn {

MetaWareNNBackend::MetaWareNNBackend() {
  LOG(INFO) << "MetaWareNNBackend constructor";
  json supported_ops;
  std::set<int32_t> supported_tflite_ops;

  std::string json_path = std::string(std::getenv("METAWARENN_LIB_PATH")) +
                          "mwnnconvert/json/supported_ops.json";

  std::ifstream ops_file{json_path};
  ops_file >> supported_ops;
  std::set<std::string> supported_ops_glow = supported_ops["glow"];
  supported_ops_ = supported_ops_glow;
}

runtime::DeviceManager * MetaWareNNBackend::createDeviceManager(
    const runtime::DeviceConfig &deviceConfig) {
  return createMetaWareNNDeviceManager(deviceConfig);
}

Expected<std::unique_ptr<CompiledFunction>> MetaWareNNBackend::compile(
    Function *F, const BackendOptions &opts) const {
  LOG(INFO) << "In MetaWareNNBackend::compile";

  std::unique_ptr<MetaWareNNFunction> compiledFunc =
      glow::make_unique<MetaWareNNFunction>(
          glow::runtime::RuntimeBundle::create(*F), F);
  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

bool MetaWareNNBackend::isOpSupported(const NodeInfo &NI) const {
  if(supported_ops_.count(NI.getKindName())) {
    return true;
  } else {
    return false;
  }
}

bool MetaWareNNBackend::shouldLower(const ::glow::Node *N) const {
  return false;
}

unsigned MetaWareNNBackend::numDevices() {
  return 1;
}

}  // namespace metawarenn
