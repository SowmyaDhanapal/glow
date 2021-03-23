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
  //Inference Part
  LOG(INFO) << "runFunction!";
  std::cout << "\nGetting graph from shared memory";
  namespace bip = boost::interprocess;
  bip::shared_memory_object shm(bip::open_only, "SharedMemoryFile", bip::read_only);
  bip::mapped_region region(shm, bip::read_only);
  bip::bufferstream bs(std::ios::in);
  bs.buffer(reinterpret_cast<char*>(region.get_address()), region.get_size());
  boost::archive::text_iarchive ia(bs);
  MWNNGraph graph;
  ia >> graph;
  std::cout << "\nCalling convert_to_mwnn_format";
  convert_to_mwnn_format(graph, CHW_TO_HWC);
  bip::shared_memory_object::remove("SharedMemoryFile");
  exit(1);
}

void MetaWareNNDeviceManager::addNetwork(const Module *module,
                                     glow::runtime::FunctionMapTy functions,
                                     glow::runtime::ReadyCBTy readyCB) {
  auto function = module->getFunctions().front();
  namespace bip = boost::interprocess;
  bip::shared_memory_object shm(bip::create_only, "SharedMemoryFile", bip::read_write);
  shm.truncate(60u<<20); // 60MiB
  bip::mapped_region region(shm, bip::read_write);
  bip::bufferstream bs(std::ios::out);
  bs.buffer(reinterpret_cast<char*>(region.get_address()), region.get_size());
  boost::archive::text_oarchive oa(bs);
  MWNNGraph mwnn_graph(function); //Parse the GLOW Function to MWNNGraph
  optimizer::PassManager manager;
  if(CHW_TO_HWC)
  {
    for (auto g_t : mwnn_graph.get_graph_inputs()) {
      if(g_t.get_dims().size() == 4) {
        std::cout << "\n Name : " << g_t.get_name();
        std::cout << "\t Dims : ";
        for (auto dim : g_t.get_dims())
          std::cout << dim << ",";
        optimizer::ConvertLayout cl(&mwnn_graph, g_t, CHW_TO_HWC, 0);
        manager.register_pass(cl);
      }
    }
  }
  auto node_list = mwnn_graph.get_graph_nodes();
  for (int node_idx = 0; node_idx < mwnn_graph.get_graph_nodes().size(); node_idx++) {
    auto g_n = node_list[node_idx];
    if(g_n.get_op_type() == "Relu") {
      optimizer::FuseRelu fr(&mwnn_graph, g_n);
      std::cout << "\n MetaWareNNCC : " << fr.get_name();
      manager.register_pass(fr);
    }
    else if(g_n.get_op_type() == "Transpose") {
      LOG(INFO) << "Inside Transpose";
      optimizer::RemoveTranspose rt(&mwnn_graph, g_n);
      std::cout << "\n MetaWareNNCC : " << rt.get_name();
      manager.register_pass(rt);
    }
  }
  manager.run_passes();
  oa << mwnn_graph;
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
