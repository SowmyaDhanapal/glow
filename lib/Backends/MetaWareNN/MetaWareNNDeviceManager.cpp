#include "MetaWareNNDeviceManager.h"

using namespace glow;
using namespace glow::runtime;

namespace metawarenn {

std::atomic<RunIdentifierTy> MetaWareNNDeviceManager::runIdentifier_;

DeviceManager *createMetaWareNNDeviceManager(const DeviceConfig &config) {
  return new MetaWareNNDeviceManager(config);
}

Error MetaWareNNDeviceManager::init() { return Error::success(); }

MetaWareNNDeviceManager::MetaWareNNDeviceManager(const DeviceConfig &config)
    : DeviceManager(config) {LOG(INFO) << "MetaWareNNDeviceManager!!!";}

MetaWareNNDeviceManager::~MetaWareNNDeviceManager() {}

RunIdentifierTy
MetaWareNNDeviceManager::runFunction(std::string functionName,
                                 std::unique_ptr<ExecutionContext> ctx,
                                 runtime::ResultCBTy resultCB) {
  //Inference Part
  LOG(INFO) << "runFunction!";
  RunIdentifierTy runId = runIdentifier_++;
  auto it = mwnn_functions_.find(functionName);
  int graph_id;
  MetaWareNNFunction *mwnn_function;
  graph_id = (it->second).graph_id;
  mwnn_function = (it->second).function;
  auto executeErr = mwnn_function->execute(ctx.get());
  if (executeErr) {
    resultCB(runId, std::move(executeErr), std::move(ctx));
    return -1;
  }
  resultCB(runId, Error::success(), std::move(ctx));
  return runId;
}

void MetaWareNNDeviceManager::addNetwork(const Module *module,
                                         glow::runtime::FunctionMapTy functions,
                                         glow::runtime::ReadyCBTy readyCB) {
  int graph_count = 0;
  for (auto &func : functions) {
    graph_count++;
    MetaWareNNFunction *mwnnFunction = static_cast<MetaWareNNFunction*>(func.second);

    // Insert the mwnnFunction into mwnn_functions_.
    bool inserted = false;
    std::tie(std::ignore, inserted) = mwnn_functions_.insert(std::make_pair(func.first,
                          MetaWareNNFunctionMeta{graph_count, mwnnFunction}));

  }
  readyCB(module, Error::success());
}

void MetaWareNNDeviceManager::evictNetwork(std::string functionName,
                                       glow::runtime::EvictFunctionCBTy evictCB) {
  DCHECK(evictCB != nullptr);
  evictCB(functionName, Error::success());
 }

uint64_t MetaWareNNDeviceManager::getMaximumMemory() const {
  return 415632456555555; //Temporary memory
}

uint64_t MetaWareNNDeviceManager::getAvailableMemory() const {
  return 415632456555555; //Temporary memory
}

bool MetaWareNNDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return true;
}

Error MetaWareNNDeviceManager::stop(bool block) {
  return Error::success();
}

} // namespace metawarenn
