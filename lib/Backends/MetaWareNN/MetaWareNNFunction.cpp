#include "MetaWareNNFunction.h"

namespace metawarenn {

MetaWareNNFunction::MetaWareNNFunction(runtime::RuntimeBundle &&bundle, Function *F)
    : CompiledFunction(std::move(bundle)) {
    findIOPlaceholders(F);
    graph_count++;
    std::string subgraph_name = "MetaWareNN_" + std::to_string(graph_count);
    mwnn_graph_ = std::make_shared<MWNNGraph>(F, subgraph_name); //Parse the GLOW Function to MWNNGraph
    optimizer::PassManager manager;
    if(CHW_TO_HWC)
    {
      for (auto g_t : mwnn_graph_->get_graph_inputs()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          optimizer::ConvertLayout cl(mwnn_graph_, g_t, CHW_TO_HWC, 0);
          manager.register_pass(cl);
        }
      }
    }
    if(HWC_TO_CHW)
    {
      for (auto g_t : mwnn_graph_->get_graph_initializers()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph_, g_t, 0, HWC_TO_CHW);
          manager.register_pass(cl);
        }
      }
      //Subgraph from other backends is already in CHW order
      if(graph_count == 1) {
        for (auto g_t : mwnn_graph_->get_graph_inputs()) {
          if(g_t.get_dims().size() == 4) {
            std::cout << "\n Name : " << g_t.get_name();
            std::cout << "\t Dims : ";
            for (auto dim : g_t.get_dims())
              std::cout << dim << ",";
            ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph_, g_t, 0, HWC_TO_CHW);
            manager.register_pass(cl);
          }
        }
      }
    }
    auto node_list = mwnn_graph_->get_graph_nodes();
    for (int node_idx = 0; node_idx < mwnn_graph_->get_graph_nodes().size(); node_idx++) {
      auto g_n = node_list[node_idx];
      if(g_n.get_op_type() == "Relu") {
        optimizer::FuseRelu fr(mwnn_graph_, g_n);
        //std::cout << "\n MetaWareNNCC : " << fr.get_name();
        manager.register_pass(fr);
      }
      else if(g_n.get_op_type() == "Transpose") {
        LOG(INFO) << "Inside Transpose";
        optimizer::RemoveTranspose rt(mwnn_graph_, g_n);
        //std::cout << "\n MetaWareNNCC : " << rt.get_name();
        manager.register_pass(rt);
      }
    }
    manager.run_passes();
    #if INVOKE_NNAC
      std::cout << "\n ---------------------------Graph----------------------------- \n";
      std::cout << "\n Graph Name : " << mwnn_graph_->get_name();
      ::MWNN::MWNNGraphProto mwnn_graph_proto;
      mwnn_graph_proto.set_name(mwnn_graph_->get_name());
      mwnn_graph_proto.set_graph_input(mwnn_graph_->get_graph_ip_name());
      mwnn_graph_proto.set_graph_output(mwnn_graph_->get_graph_op_name());

      //std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
      for (auto g_ip : mwnn_graph_->get_graph_inputs()) {
        /*std::cout << "\n Input Name : " << g_ip.get_name();
        std::cout << "\n Data Type : " << g_ip.get_type();
        std::cout << "\n Input Dims : ";*/
        auto input = mwnn_graph_proto.add_input();
        input->set_name(g_ip.get_name());
        input->set_type(g_ip.get_type());

        for (auto dim : g_ip.get_dims())
          input->add_dims(dim);
      }
      //std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
      for (auto g_op : mwnn_graph_->get_graph_outputs()) {
        /*std::cout << "\n Output Name : " << g_op.get_name();
        std::cout << "\n Data Type : " << g_op.get_type();
        std::cout << "\n Output Dims : ";*/
        auto output = mwnn_graph_proto.add_output();
        output->set_name(g_op.get_name());
        output->set_type(g_op.get_type());
        for (auto dim : g_op.get_dims())
          output->add_dims(dim);

      }
      //std::cout << "\n -----------------------Graph Nodes-------------------------- \n";
      for (auto g_n : mwnn_graph_->get_graph_nodes()) {
        /*std::cout << "\n ================================================================ \n";
        std::cout << "\n Node Name : " << g_n.get_name();
        std::cout << "\n Op Type : " << g_n.get_op_type();*/
        auto node = mwnn_graph_proto.add_node();
        node->set_name(g_n.get_name());
        auto op_type = g_n.get_op_type();
        node->set_op_type(op_type == "DepthwiseConv" ? "Conv" : op_type);
        for (auto n_ip : g_n.get_inputs()) {
          //std::cout << "\n Input : n_ip : " << n_ip;
          node->add_input((n_ip));
        }
        for (auto n_op : g_n.get_outputs()) {
          //std::cout << "\n Output : n_op : " << n_op;
          node->add_output((n_op));
        }
        //std::cout << "\n ---------------------------------------------------------------- ";
        for (auto attribute : g_n.get_attributes()) {
          /*std::cout << "\n Attribute Name : " << attribute.get_name();
          std::cout << "\n Attribute Data Type : " << attribute.get_type();
          std::cout << "\n Attribute Data : ";*/
          auto attr = node->add_attribute();
          attr->set_name(attribute.get_name());
          attr->set_type(attribute.get_type());
          if(attribute.get_type() == 3 || attribute.get_type() == 8) {
            for(int i = 0; i < attribute.get_string_data().size(); i++){
              auto data = attr->add_data();
              data = &attribute.get_string_data()[i];
              //std::cout << attribute.get_string_data()[i] << ",";
            }
          }
          else {
            for(int i = 0; i < attribute.get_data().size(); i++) {
              //std::cout << attribute.get_data()[i] << ",";
              attr->add_ints(attribute.get_data()[i]);
            }
          }
        }
      }
      //std::cout << "\n -----------------------Graph Tensors-------------------------- \n";
      for (auto g_t : mwnn_graph_->get_graph_initializers()) {
        auto initializer = mwnn_graph_proto.add_initializer();
        initializer->set_name(g_t.get_name());
        initializer->set_data_type(g_t.get_type());
        /*std::cout << "\n Name : " << g_t.get_name();
        std::cout << "\n Type : " << g_t.get_type();
        std::cout << "\n Dim : ";*/
        for (auto dim : g_t.get_dims()){
          //std::cout << dim << ",";
          initializer->add_dims(dim);
        }
        for (auto t_val : g_t.get_tensor())
          initializer->add_float_data(t_val);
      }

      std::cout << "\n Graph Name : " << mwnn_graph_->get_name();
      std::string g_name = mwnn_graph_->get_name();
      auto mwnn_op_path = "/path/to/store/mwnn_dump_files/";
      auto mwnn_proto_bin = std::string(mwnn_op_path) + std::string(g_name) + ".bin";

      int fp = open(mwnn_proto_bin.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      std::cout << fp;
      std::cout << mwnn_graph_proto.SerializeToFileDescriptor(fp);
      close(fp);

      std::cout << "\n\n=================Initiating NNAC python script via shell script======================\n";
      std::string cmd = "bash /path/to/glow/lib/Backends/MetaWareNN/metawarenn_lib/mwnnconvert/mwnn_convert.sh " + mwnn_proto_bin + " " + mwnn_op_path + " " + g_name + " " + std::to_string(graph_count);
      const char *command = cmd.c_str();
      system(command);
      std::ifstream in;
      in.open("/path/to/ARC/cnn_tools/utils/tools/evgencnn/scripts/cnn_bin_" + g_name + ".bin", std::ios::in | std::ios::binary);
      if(in.is_open())
      {
          std::streampos start = in.tellg();
          in.seekg(0, std::ios::end);
          std::streampos end = in.tellg();
          in.seekg(0, std::ios::beg);
          std::vector<char> contents;
          contents.resize(static_cast<size_t>(end - start));
          in.read(&contents[0], contents.size());
          auto data = contents.data();
          int shmid;
          struct shmseg *shmp;
          //Create unique key
          key_t key = ftok("/tmp/", graph_count);
          std::cout << "\nkey: " << key;
          shmid = shmget(key, sizeof(struct shmseg), 0644|IPC_CREAT);
          if (shmid == -1) {
              perror("Shared memory");
              exit(1);
          }
          shmp = (shmseg*)shmat(shmid, NULL, 0);
          if (shmp == (void *) -1) {
              perror("Shared memory attach");
              exit(1);
          }
          memcpy(shmp->buf, data, contents.size());
          shmp->cnt = contents.size();
          printf("\nWriting Process: Shared Memory Write: Wrote %d bytes\n", shmp->cnt);
          sleep(3);
          printf("\nWriting Process: Complete\n");
      }
    #endif
}
void MetaWareNNFunction::findIOPlaceholders(Function *F) {
  for (auto const &V : F->getParent()->getPlaceholders()) {
    if (!usedInFunction(V, F)) {
      continue;
    }
    if (getOutputSave(F, V)) {
      std::cout << "\n V->getName() in *88888 if op: " << std::string(V->getName());
      outputs_.push_back(V);
    } else {
      std::cout << "\n V->getName() in *88888 elee: " << std::string(V->getName());
      inputs_.push_back(V);
    }
  }
}

MetaWareNNFunction::~MetaWareNNFunction() {}

Error MetaWareNNFunction::execute(ExecutionContext *context) {
  //Fills the graph_inputs with input data pointer using indexes
  std::unordered_map<std::string, float*> graph_inputs;
  std::unordered_map<std::string, float*> graph_outputs;
  auto bindings = context->getPlaceholderBindings();
  for (const auto &ph : this->getInputs()) {
    auto *tensor = bindings->get(ph);
    graph_inputs[std::string(ph->getName())] = (float*)tensor->getUnsafePtr();
  }
  for (const auto &ph : this->getOutputs()) {
    auto *tensor = bindings->get(ph);
    graph_outputs[mwnn_graph_->get_graph_op_name()] = (float*)tensor->getUnsafePtr();

  }
  convert_to_mwnn_format(*mwnn_graph_, graph_inputs, graph_outputs, CHW_TO_HWC);
  return Error::success();
}
} // namespace metawarenn
