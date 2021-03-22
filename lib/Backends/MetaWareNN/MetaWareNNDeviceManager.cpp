#include "MetaWareNNDeviceManager.h"

using namespace glow;
using namespace glow::runtime;

namespace metawarenn {

DeviceManager *createMetaWareNNDeviceManager(const DeviceConfig &config) {
  return new MetaWareNNDeviceManager(config);
}

MetaWareNNDeviceManager::MetaWareNNDeviceManager(const DeviceConfig &config)
    : DeviceManager(config) {LOG(INFO) << "MetaWareNNDeviceManager!!!";}

MetaWareNNDeviceManager::~MetaWareNNDeviceManager() {}

RunIdentifierTy
MetaWareNNDeviceManager::runFunction(std::string functionName,
                                 std::unique_ptr<ExecutionContext> ctx,
                                 runtime::ResultCBTy resultCB)
{
  LOG(INFO) << "runFunction!";
  exit(1);
}

void MetaWareNNDeviceManager::addNetwork(const Module *module,
                                     glow::runtime::FunctionMapTy functions,
                                     glow::runtime::ReadyCBTy readyCB) {
  readyCB(module, Error::success());
}

void MetaWareNNDeviceManager::evictNetwork(std::string functionName,
                                       glow::runtime::EvictFunctionCBTy evictCB) {}
                                    
uint64_t MetaWareNNDeviceManager::getMaximumMemory() const {
  return 415632456555555; //Temporary memory
}

uint64_t MetaWareNNDeviceManager::getAvailableMemory() const {
  return 415632456555555; //Temporary memory
}

bool MetaWareNNDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return true;
}                                                             

} //namespace metawarenn