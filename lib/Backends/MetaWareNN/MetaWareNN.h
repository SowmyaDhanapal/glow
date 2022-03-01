#ifndef GLOW_METAWARENN_BACKEND_H
#define GLOW_METAWARENN_BACKEND_H

#include "glow/Backend/Backend.h"
#include "MetaWareNNDeviceManager.h"

using namespace glow;

namespace metawarenn {

class MetaWareNNBackend final : public Backend {
 public:
  MetaWareNNBackend();
  ~MetaWareNNBackend() override = default;
  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "MetaWareNN"; }
  static unsigned numDevices();
  // Creates MetaWareNN Device manager with passed config
  runtime::DeviceManager * createDeviceManager(
      const runtime::DeviceConfig &deviceConfig) override;
  // Compile function is just used to maintain the class override.
  Expected<std::unique_ptr<CompiledFunction>> compile(
    Function *F, const BackendOptions &opts) const override;
  // Used to check if an operator is supported by MetaWareNN Backend
  bool isOpSupported(const NodeInfo &NI) const override;
  bool shouldLower(const ::glow::Node *N) const override;
};

}  // namespace metawarenn
#endif  // GLOW_METAWARENN_BACKEND_H