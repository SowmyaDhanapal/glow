#ifndef GLOW_BACKENDS_METAWARENN_METAWARENNDEVICEMANAGER_H
#define GLOW_BACKENDS_METAWARENN_METAWARENNDEVICEMANAGER_H

#include "glow/Backends/DeviceManager.h"
#include "glow/Graph/Utils.h"
#include "MetaWareNNFunction.h"

using namespace glow;
using namespace glow::runtime;

namespace metawarenn {

class MetaWareNNDeviceManager : public glow::runtime::DeviceManager {

public:
  MetaWareNNDeviceManager(const glow::runtime::DeviceConfig &config);
  ~MetaWareNNDeviceManager();
  glow::runtime::RunIdentifierTy
  runFunction(std::string functionName,
                                  std::unique_ptr<glow::ExecutionContext> ctx,
                                  runtime::ResultCBTy resultCB) override;
  void addNetwork(const Module *module,
                                      glow::runtime::FunctionMapTy functions,
                                      glow::runtime::ReadyCBTy readyCB) override;
  void evictNetwork(std::string functionName,
                                        glow::runtime::EvictFunctionCBTy evictCB) override;
  uint64_t getMaximumMemory() const override;
  uint64_t getAvailableMemory() const override;
  bool isMemoryAvailable(uint64_t estimate) const override;
  Error init() override;
  Error stop(bool block) override;
private:
  struct MetaWareNNFunctionMeta {
    MetaWareNNFunction *function;
  };
  std::unordered_map<std::string, MetaWareNNFunctionMeta> mwnn_functions_;
  static std::atomic<glow::runtime::RunIdentifierTy> runIdentifier_;
};
  glow::runtime::DeviceManager *createMetaWareNNDeviceManager(const glow::runtime::DeviceConfig &config);

} // namespace metawarenn

#endif // GLOW_BACKENDS_HABANADEVICEMANAGER_H
