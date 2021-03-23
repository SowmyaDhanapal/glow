#ifndef GLOW_BACKENDS_METAWARENN_METAWARENNDEVICEMANAGER_H
#define GLOW_BACKENDS_METAWARENN_METAWARENNDEVICEMANAGER_H

#include "glow/Backends/DeviceManager.h"
#include "metawarenn_lib/metawarenn_graph.h"
#include "glow/Graph/Utils.h"

#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/optimizer/metawarenn_optimizer.h"
#include "metawarenn_lib/metawarenn_utils.h"

#define CHW_TO_HWC 1

using namespace glow;

namespace metawarenn {

class MetaWareNNDeviceManager : public  glow::runtime::DeviceManager {

public:
  MetaWareNNDeviceManager(const  glow::runtime::DeviceConfig &config);
  ~MetaWareNNDeviceManager();
  glow::runtime::RunIdentifierTy
  runFunction(std::string functionName,
                                  std::unique_ptr<ExecutionContext> ctx,
                                  runtime::ResultCBTy resultCB) override;
  void addNetwork(const Module *module,
                                      glow::runtime::FunctionMapTy functions,
                                      glow::runtime::ReadyCBTy readyCB) override;
  void evictNetwork(std::string functionName,
                                        glow::runtime::EvictFunctionCBTy evictCB) override;
  uint64_t getMaximumMemory() const override;
  uint64_t getAvailableMemory() const override;
  bool isMemoryAvailable(uint64_t estimate) const override;

private:
  MWNNGraph mwnn_graph_;
};
  glow::runtime::DeviceManager *createMetaWareNNDeviceManager(const glow::runtime::DeviceConfig &config);
}

#endif // GLOW_BACKENDS_HABANADEVICEMANAGER_H