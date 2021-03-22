#ifndef GLOW_METAWARENN_BACKEND_H
#define GLOW_METAWARENN_BACKEND_H

#include "glow/Backend/Backend.h"
#include "DummyFunction.h"

using namespace glow;

namespace metawarenn {

class MetaWareNNBackend final : public Backend {
public:
  MetaWareNNBackend();
  ~MetaWareNNBackend() override = default;
  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "MetaWareNN"; }
  static unsigned numDevices();
  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;
  bool isOpSupported(const NodeInfo &NI) const override;
  bool shouldLower(const Node *N) const override;
};

} // namespace metawarenn
#endif // GLOW_METAWARENN_BACKEND_H