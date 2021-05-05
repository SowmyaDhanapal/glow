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
  bip::shared_memory_object::remove("SharedMemoryFile");
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
  if(HWC_TO_CHW)
  {
    for (auto g_t : mwnn_graph.get_graph_initializers()) {
      if(g_t.get_dims().size() == 4) {
        std::cout << "\n Name : " << g_t.get_name();
        std::cout << "\t Dims : ";
        for (auto dim : g_t.get_dims())
          std::cout << dim << ",";
        ::metawarenn::optimizer::ConvertLayout cl(&mwnn_graph, g_t, 0, HWC_TO_CHW);
        manager.register_pass(cl);
      }
    }
    for (auto g_t : mwnn_graph.get_graph_inputs()) {
      if(g_t.get_dims().size() == 4) {
        std::cout << "\n Name : " << g_t.get_name();
        std::cout << "\t Dims : ";
        for (auto dim : g_t.get_dims())
          std::cout << dim << ",";
        ::metawarenn::optimizer::ConvertLayout cl(&mwnn_graph, g_t, 0, HWC_TO_CHW);
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
  #if INVOKE_NNAC
    std::cout << "\n ---------------------------Graph----------------------------- \n";
    std::cout << "\n Graph Name : " << mwnn_graph.get_name();
    ::MWNN::MWNNGraphProto mwnn_graph_proto;
    mwnn_graph_proto.set_name(mwnn_graph.get_name());
    mwnn_graph_proto.set_graph_input(mwnn_graph.get_graph_ip_name());
    mwnn_graph_proto.set_graph_output(mwnn_graph.get_graph_op_name());

    std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
    for (auto g_ip : mwnn_graph.get_graph_inputs()) {
      std::cout << "\n Input Name : " << g_ip.get_name();
      std::cout << "\n Data Type : " << g_ip.get_type();
      std::cout << "\n Input Dims : ";
      auto input = mwnn_graph_proto.add_input();
      input->set_name(g_ip.get_name());
      input->set_type(g_ip.get_type());

      for (auto dim : g_ip.get_dims())
        input->add_dims(dim);
    }
    std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
    for (auto g_op : mwnn_graph.get_graph_outputs()) {
      std::cout << "\n Output Name : " << g_op.get_name();
      std::cout << "\n Data Type : " << g_op.get_type();
      std::cout << "\n Output Dims : ";
      auto output = mwnn_graph_proto.add_output();
      output->set_name(g_op.get_name());
      output->set_type(g_op.get_type());
      for (auto dim : g_op.get_dims())
        output->add_dims(dim);

    }
    std::cout << "\n -----------------------Graph Nodes-------------------------- \n";
    for (auto g_n : mwnn_graph.get_graph_nodes()) {
      std::cout << "\n ================================================================ \n";
      std::cout << "\n Node Name : " << g_n.get_name();
      std::cout << "\n Op Type : " << g_n.get_op_type();
      auto node = mwnn_graph_proto.add_node();
      node->set_name(g_n.get_name());
      auto op_type = g_n.get_op_type();
      node->set_op_type(op_type == "DepthwiseConv" ? "Conv" : op_type);
      for (auto n_ip : g_n.get_inputs())
        node->add_input((n_ip));
      for (auto n_op : g_n.get_outputs())
        node->add_output((n_op));
      std::cout << "\n ---------------------------------------------------------------- ";
      for (auto attribute : g_n.get_attributes()) {
        std::cout << "\n Attribute Name : " << attribute.get_name();
        std::cout << "\n Attribute Data Type : " << attribute.get_type();
        std::cout << "\n Attribute Data : ";
        auto attr = node->add_attribute();
        attr->set_name(attribute.get_name());
        attr->set_type(attribute.get_type());
        if(attribute.get_type() == 3 || attribute.get_type() == 8) {
          for(int i = 0; i < attribute.get_string_data().size(); i++){
            auto data = attr->add_data();
            data = &attribute.get_string_data()[i];
            std::cout << attribute.get_string_data()[i] << ",";}
        }
        else {
          for(int i = 0; i < attribute.get_data().size(); i++){
            std::cout << attribute.get_data()[i] << ",";
            attr->add_ints(attribute.get_data()[i]);
          }
        }
      }
    }
    std::cout << "\n -----------------------Graph Tensors-------------------------- \n";
    for (auto g_t : mwnn_graph.get_graph_initializers()) {
      auto initializer = mwnn_graph_proto.add_initializer();
      initializer->set_name(g_t.get_name());
      initializer->set_data_type(g_t.get_type());
      std::cout << "\n Name : " << g_t.get_name();
      std::cout << "\n Type : " << g_t.get_type();
      std::cout << "\n Dim : ";
      for (auto dim : g_t.get_dims()){
        std::cout << dim << ",";
        initializer->add_dims(dim);
      }
      for (auto t_val : g_t.get_tensor())
        initializer->add_float_data(t_val);
    }
    int fp = open("/path/to/store/mobilenetv2-7_graphproto.bin", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::cout << fp;
    std::cout << mwnn_graph_proto.SerializeToFileDescriptor(fp);
    close(fp);

    std::cout << "\n\n=======================Initiating NNAC python script=================================\n";
    system("bash /path/to/glow/lib/Backends/MetaWareNN/metawarenn_lib/mwnnconvert/mwnn_convert.sh");
    exit(1);
  #endif
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
