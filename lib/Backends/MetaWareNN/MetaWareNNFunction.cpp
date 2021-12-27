#include "MetaWareNNFunction.h"

namespace metawarenn {

std::set<Kinded::Kind> onnx_unsupported_nodes = {Kinded::Kind::ChannelShuffleNodeKind};

// Converts the ONNX unsupported nodes in GLOW to modified ops and add it in GLOW function
void convert_glow_to_onnx(Function *F)
{
  GraphPostOrderVisitor visitor(*F);
  auto node_list = visitor.getPostOrder();
  for (auto *node : node_list) {
    if(onnx_unsupported_nodes.count(node->getKind())) {
      switch (node->getKind())
      {
        //ChannelShuffle -> Reshape + Transpose + Reshape
        case Kinded::Kind::ChannelShuffleNodeKind:
        {
          auto *channel_shuffle_node = llvm::dyn_cast<ChannelShuffleNode>(node);
          auto input = channel_shuffle_node->getInput();
          auto group = channel_shuffle_node->getGroup();
          auto kernel = channel_shuffle_node->getKernel();
          auto inDims = input.dims();

          ShapeVector dims(inDims.begin(), inDims.end());
          auto D = dims[kernel];
          assert(D % group == 0);

          dims.erase(dims.begin() + kernel);
          dims.insert(dims.begin() + kernel, D / group);
          dims.insert(dims.begin() + kernel, group);
          bool update_layout = false;
          // Check if dimenension is in NHWC, if so update the layout in reshape layer
          if(!(dims[dims.size() - 1] == dims[dims.size() - 2]))
            update_layout = true;
          // Update the Shape Order for NHWC to NCHW
          ReshapeNode *R1;
          TransposeNode *T;
          ReshapeNode *R2;

          if(update_layout) {
          ShapeVector new_dims(dims.size());
          new_dims[1] = dims[3];
          new_dims[2] = dims[4];
          new_dims[3] = dims[1];
          new_dims[4] = dims[2];
          new_dims[0] = dims[0];
          R1 = F->createReshape(channel_shuffle_node->getName().str() + "_reshape1", input, new_dims);
          }
          else
            R1 = F->createReshape(channel_shuffle_node->getName().str() + "_reshape1", input, dims);
          std::cout << "\nlayout: " << R1->getLayout().c_str();
          std::vector<unsigned_t> transpose(dims.size());
          for (size_t i = 0; i < transpose.size(); i++) {
            transpose[i] = i;
          }
          // Update the Permutation Order for NHWC to NCHW
          std::vector<unsigned_t> new_transpose(transpose.begin(), transpose.end());
          new_transpose[1] = transpose[2];
          new_transpose[2] = transpose[1];
          T = F->createTranspose(channel_shuffle_node->getName().str() + "_transpose", R1,
                                      new_transpose, R1->getLayout());
          // Update the Shape Order for NHWC to NCHW
          if(update_layout) {
          ShapeVector new_in_dims(inDims.begin(), inDims.end());
          new_in_dims[1] = inDims[3];
          new_in_dims[3] = inDims[1];
          R2 = F->createReshape(channel_shuffle_node->getName().str() + "_reshape2", T, new_in_dims,
                                      T->getLayout());
          }
          else
            R2 = F->createReshape(channel_shuffle_node->getName().str() + "_reshape2", T, inDims,
                                      T->getLayout());
          channel_shuffle_node->getResult().replaceAllUsesOfWith(R2->getResult());
          F->eraseNode(channel_shuffle_node); // Remove the ChannelShuffle Node from GLOW Function
        }
      }
    }
  }
}

template<class T1, class T2>
void MetaWareNNFunction::read_tensor(glow::Constant *c, std::string tensor_name, ElemKind elem_kind) {
  std::vector<int> const_dims(c->dims().size());
  std::cout << "\n initializer - " << tensor_name << ": ";
  int i = 0;
  for(auto dim: c->dims()) {
    const_dims[i++] = (int)dim;
  }
  for(auto dim: const_dims)
    std::cout << dim << ", ";

  auto size = std::accumulate(begin(const_dims), end(const_dims), 1, std::multiplies<double>());
  glow::Handle<T1> handle = c->getHandle<T1>();
  auto begin = &handle.raw(0);
  std::vector<T2> data(begin, begin + handle.actualSize());
  metawarenn::Tensor tensor(tensor_name, const_dims, get_mwnn_type_glow(elem_kind), data);
  graph_->set_graph_initializers(tensor);
  graph_->initializer_names.insert(tensor.get_name());
}

void MetaWareNNFunction::CreateMWNNNode(const std::string &node_name_,
                                  const std::string &node_op_type_,
                                  const std::vector<::metawarenn::Attribute> &node_attributes_,
                                  const std::vector<std::string> &node_inputs_,
                                  const std::vector<std::string> &node_outputs_) {
  ::metawarenn::Node m_node(node_name_, node_op_type_, node_attributes_, node_inputs_, node_outputs_);
  graph_->set_graph_nodes(m_node);

  std::cout << "\n ================================Node=============================\n";
  std::cout << "\n Name : " << node_name_;
  std::cout << "\n Type : " << node_op_type_;
  for (auto nip: node_inputs_)
    std::cout << "\n Inputs : " << nip;
  for (auto nop: node_outputs_)
    std::cout << "\n Outputs : " << nop;
}

void MetaWareNNFunction::CreateMWNNQuantParams(NodeValue c, std::string tensor_name) {
  std::string scale_name = tensor_name + std::string("_scale");
  std::vector<float> tensor_vec_scale = {c.getScale()};
  ::metawarenn::Tensor scale_tensor(scale_name, std::vector<int>({tensor_vec_scale.size()}),
                            ::metawarenn::ElementType::element_type::float_, tensor_vec_scale);
  graph_->set_graph_initializers(scale_tensor);
  graph_->initializer_names.insert(scale_name);

  std::string zp_name = tensor_name + std::string("_zero_point");
  std::vector<int32_t> tensor_vec_zp = {c.getOffset()};
  ElementType::element_type type;
  if(c.getElementType() == ElemKind::FloatTy)
    type = ElementType::element_type::uint8_;
  else
    type = get_mwnn_type_glow(c.getElementType());
  ::metawarenn::Tensor zp_tensor(zp_name, std::vector<int>({tensor_vec_zp.size()}),
                            type, tensor_vec_zp);
  graph_->set_graph_initializers(zp_tensor);
  graph_->initializer_names.insert(zp_name);
}

void MetaWareNNFunction::CreateQDQNodes(std::string ip_name, std::string op_name,
                      std::string node_name) {
  std::string quant_node_op_type = "QuantizeLinear";
  std::string quant_node_name = quant_node_op_type + "_" + ip_name;
  std::vector<std::string> quant_node_inputs;
  std::vector<std::string> quant_node_outputs;
  std::vector<::metawarenn::Attribute> quant_node_attributes;
  quant_node_inputs.push_back(ip_name);
  quant_node_inputs.push_back(node_name + "_scale");//Output Scale
  quant_node_inputs.push_back(node_name + "_zero_point");//Output ZeroPoint
  quant_node_outputs.push_back(quant_node_name);
  // Create QuantizeLinear node with quantization params
  CreateMWNNNode(quant_node_name, quant_node_op_type, quant_node_attributes, quant_node_inputs, quant_node_outputs);

  std::string dequant_node_op_type = "DequantizeLinear";
  std::string dequant_node_name = dequant_node_op_type + "_" + ip_name;
  std::vector<std::string> dequant_node_inputs;
  std::vector<std::string> dequant_node_outputs;
  std::vector<::metawarenn::Attribute> dequant_node_attributes;
  dequant_node_inputs.push_back(quant_node_outputs[0]);
  dequant_node_inputs.push_back(node_name + "_scale");//Output Scale
  dequant_node_inputs.push_back(node_name + "_zero_point");//Output ZeroPoint
  dequant_node_outputs.push_back(op_name);
  // Create DequantizeLinear node with quantization params
  CreateMWNNNode(dequant_node_name, dequant_node_op_type, dequant_node_attributes, dequant_node_inputs, dequant_node_outputs);
}

MetaWareNNFunction::MetaWareNNFunction(runtime::RuntimeBundle &&bundle, Function *F)
    : CompiledFunction(std::move(bundle)) {
    findIOPlaceholders(F);
    graph_count++;
    convert_glow_to_onnx(F);
    std::string subgraph_name = "MetaWareNN_" + std::to_string(graph_count);

    /*Create MetaWareNN High Level Graph Representation from Glow SubGraph Function*/
    graph_ = std::make_shared<Graph>();
    graph_->set_name(subgraph_name);

    std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
    std::cout << "\n MWNN Graph Name : " << graph_->get_name();

    GraphPostOrderVisitor visitor(*F);
    auto node_list = visitor.getPostOrder();
    std::string global_input_name;
    std::string global_output_name = "";
    std::set<std::string> ignored_transpose_nodes;
    int transpose_cnt = 0;
    std::set<std::string> node_inputs_map;
    std::map<std::string, std::string> quant_ip_mapper;
    std::map<std::string, std::string> quant_op_mapper;
    std::vector<std::string> node_inputs;
    std::vector<std::string> node_outputs;
    std::vector<::metawarenn::Attribute> node_attributes;
    int node_index = 0;
    // Graph input and output handling
    auto remove_transpose = false;
    auto quantize_encounter = false;
    for (auto &V : F->getParent()->getPlaceholders()) {
      if (!usedInFunction(V, F)) {
        continue;
      }
      auto glow_dims = V->getType()->dims();
      auto data_type = V->getType()->getElementType();
      int size = glow_dims.size();
      std::vector<int> dims(size);
      int i = 0;
      for(auto dim: glow_dims)
        dims[i++] = (int(dim));
      // Input dims from NHWC to HCHW
      if(size == 4) {
        if(glow_dims[2] == glow_dims[3])
          remove_transpose = true; // Input already in NCHW, enable the first transpose removal
        else {
          dims[1] = int(glow_dims[3]);
          dims[3] = int(glow_dims[1]);
          dims[2] = int(glow_dims[2]);
          dims[0] = int(glow_dims[0]);
        }
      }
      if(V->getName().equals(F->findPlaceholders().front()->getName())) {
        std::string input_name = V->getName();
        graph_->set_graph_ip_names(input_name);
        //Fills Graph Input Tensor Details - Name, Dims
        Tensor ip_tensor(input_name, get_mwnn_type_glow(data_type), dims);
        graph_->set_graph_ip_tensor(ip_tensor);
        if(V->getElementType() == ElemKind::Int8QTy || V->getElementType() == ElemKind::UInt8QTy) {
          quantize_encounter = true;
          // Fill Scale, Zero Point initializer for graph input node
          CreateMWNNQuantParams(V->getNthResult(0), input_name);
          node_inputs.clear(); node_outputs.clear();
          std::string node_op_type = "DequantizeLinear";
          auto dequant_node_name = node_op_type + "_" + input_name;
          node_inputs.push_back(input_name);
          node_inputs.push_back(input_name + "_scale");
          node_inputs.push_back(input_name + "_zero_point");
          node_outputs.push_back(dequant_node_name);
          CreateMWNNNode(dequant_node_name, node_op_type, node_attributes, node_inputs, node_outputs);
          quant_ip_mapper[input_name] = node_outputs[0];
        }
      }
      else if (auto *save = getOutputSave(F, V)) {
        std::string output_name = save->getInput().getNode()->getName();
        if(quantize_encounter) {
          // Fill Scale, Zero Point initializer for graph output node
          CreateMWNNQuantParams(save->getInput(), output_name);
          node_inputs.clear(); node_outputs.clear();
          std::string node_op_type = "QuantizeLinear";
          auto quant_node_name = node_op_type + "_" + output_name;
          node_inputs.push_back(output_name);
          node_inputs.push_back(output_name + std::string("_scale"));
          node_inputs.push_back(output_name + std::string("_zero_point"));
          node_outputs.push_back(quant_node_name);
          CreateMWNNNode(quant_node_name, node_op_type, node_attributes, node_inputs, node_outputs);
          quant_op_mapper[output_name] = node_outputs[0];

          graph_->set_graph_op_names(node_outputs[0]);
          //Fills Graph Output Tensor Details - Name, Dims
          ::metawarenn::Tensor m_op_tensor(node_outputs[0], ElementType::element_type::uint8_, dims);
          graph_->set_graph_op_tensor(m_op_tensor);
        }
        else {
        graph_->set_graph_op_names(output_name);
        //Fills Graph Output Tensor Details - Name, Dims
        Tensor op_tensor(output_name, get_mwnn_type_glow(data_type), dims);
        graph_->set_graph_op_tensor(op_tensor);
        }
      }
    }
    for (auto *node : node_list) {
        std::cout << "\nnode_index: " << node_index;
        LOG(INFO) << "==============================================================================================================";
        std::string node_name;
        std::string node_op_type;
        node_inputs.clear();
        node_outputs.clear();
        node_attributes.clear();
        node_name = std::string(node->getName());
        auto kind = node->getKindName();
        LOG(INFO) << "Node Name: " << node_name << "\tOp type: " << node->getKindName();
        std::vector<std::string> non_op_types;
        non_op_types = {"Constant", "Placeholder", "Save"};
        if((std::count(non_op_types.begin(), non_op_types.end(), kind)) < 1)
        {
        // Node Inputs handling
        for(int i = 0; i < node->getNumInputs(); i++) {
          if((kind == "SoftMax" && i >= 1) || kind == "RescaleQuantized")
            continue;
          auto input = node->getNthInput(i);
          std::string name = input.getNode()->getName();
          if (Constant *c = llvm::dyn_cast<Constant>(input.getNode())) {
            switch(c->getElementType()) {
              case ElemKind::FloatTy: {
                read_tensor<float, float>(c, name, c->getElementType());
                break;
              }
              case ElemKind::Int8QTy:
              case ElemKind::UInt8QTy:
              case ElemKind::Int32QTy: {
                if (c->getElementType() == ElemKind::Int8QTy)
                  read_tensor<int8_t, int32_t>(c, name, c->getElementType());
                else if (c->getElementType() == ElemKind::UInt8QTy)
                  read_tensor<uint8_t, int32_t>(c, name, c->getElementType());
                else if (c->getElementType() == ElemKind::Int32QTy)
                  read_tensor<int32_t, int32_t>(c, name, c->getElementType());
                node_op_type = "DequantizeLinear";
                auto dequant_node_name = node_op_type + "_" + name;
                node_inputs.clear(); node_outputs.clear();
                CreateMWNNQuantParams(input, name);
                node_inputs.emplace_back(name);
                node_inputs.emplace_back(name + "_scale");
                node_inputs.emplace_back(name + "_zero_point");
                node_outputs.push_back(dequant_node_name);
                // Create DequantizeLinear node for initializers
                CreateMWNNNode(dequant_node_name, node_op_type, node_attributes, node_inputs, node_outputs);
                quant_ip_mapper[name] = node_outputs[0];
                break;
              }
            }
          }
          std::cout << "\n input name: " << (std::string)name;
        }
        node_inputs.clear(); node_outputs.clear();
        // Node Outputs handling
        for (int i = 0; i < node->getNumResults(); ++i) {
          if((kind == "MaxPool" && i >= 1) || kind == "RescaleQuantized")
            continue;
          auto output = node->getNthResult(i);
          std::string name = output.getNode()->getName();
          auto itr = quant_op_mapper.find(output.getNode()->getName());
          // Skip QDQ nodes for last op node and dequantize node
          if(quantize_encounter && itr == quant_op_mapper.end() && node->getKindName() != "Dequantize") {
            node_op_type = "DequantizeLinear";
            auto dequant_node_name = node_op_type + "_" + name;
            CreateMWNNQuantParams(output, name);
            CreateQDQNodes(name, dequant_node_name, name);
            quant_ip_mapper[name] = dequant_node_name;
          }
        }
        node_inputs.clear(); node_outputs.clear();
        for(int i = 0; i < node->getNumInputs(); i++) {
          auto input = node->getNthInput(i);
          auto itr = quant_ip_mapper.find(input.getNode()->getName());
          if(itr != quant_ip_mapper.end())
            node_inputs.emplace_back(itr->second);
          else
            node_inputs.emplace_back(input.getNode()->getName());
        }
        for(int i = 0; i < node->getNumResults(); i++) {
          if((kind == "MaxPool" && i >= 1))
            continue;
          auto output = node->getNthResult(i);
          node_outputs.emplace_back(output.getNode()->getName());
        }
        switch (node->getKind())
        {
          case Kinded::Kind::ConvolutionNodeKind:
          {
            node_op_type = "Conv";
            auto *conv_node = llvm::cast<ConvolutionNode>(node);
            auto dilations = conv_node->getDilation();
            auto strides = conv_node->getStrides();
            auto pads = conv_node->getPads();
            auto group = conv_node->getGroup();
            auto kernel_shape = conv_node->getKernels();
            metawarenn::Attribute attr_dilate("dilations", std::vector<int64_t>{int64_t(dilations[0]), int64_t(dilations[1])});
            node_attributes.emplace_back(attr_dilate);
            metawarenn::Attribute attr_group("group", (int64_t)group);
            node_attributes.emplace_back(attr_group);
            metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>{(int64_t)kernel_shape[0], (int64_t)kernel_shape[1]});
            node_attributes.emplace_back(attr_kernel_shape);
            metawarenn::Attribute attr_pad("pads", std::vector<int64_t>{int64_t(pads[0]), int64_t(pads[1]), int64_t(pads[2]), int64_t(pads[3])});
            node_attributes.emplace_back(attr_pad);
            metawarenn::Attribute attr_stride("strides", std::vector<int64_t>{int64_t(strides[0]), int64_t(strides[1])});
            node_attributes.emplace_back(attr_stride);
            break;
          }
          case Kinded::Kind::ReluNodeKind:
          {
            node_op_type = "Relu";
            break;
          }
        case Kinded::Kind::AvgPoolNodeKind:
        {
          auto *avgpool_node = llvm::cast<AvgPoolNode>(node);
          auto kernels = avgpool_node->getKernels();
          auto strides = avgpool_node->getStrides();
          auto pads = avgpool_node->getPads();
          auto input_dims = avgpool_node->getInput().dims();
          if(kernels[0] == input_dims[1] && kernels[1] == input_dims[2]) // Match the layer input's HW to kernel's HW
            node_op_type = "GlobalAveragePool";
          else {
            node_op_type = "AveragePool";
            auto count_include_pad = avgpool_node->getCountIncludePads();
            metawarenn::Attribute attr_count_include_pad("count_include_pad", (int64_t)count_include_pad);
            node_attributes.emplace_back(attr_count_include_pad);
            metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>{int64_t(kernels[0]), int64_t(kernels[1])});
            node_attributes.emplace_back(attr_kernel_shape);
            metawarenn::Attribute attr_stride("strides", std::vector<int64_t>{int64_t(strides[0]), int64_t(strides[1])});
            node_attributes.emplace_back(attr_stride);
            metawarenn::Attribute attr_pads("pads", std::vector<int64_t>{int64_t(pads[0]), int64_t(pads[1]), int64_t(pads[2]), int64_t(pads[3])});
            node_attributes.emplace_back(attr_pads);
          }
          break;
        }
        case Kinded::Kind::AddNodeKind:
        {
          node_op_type = "Add";
          break;
        }
        case Kinded::Kind::TransposeNodeKind:
        {
          node_op_type = "Transpose";
          auto *transpose_node = llvm::cast<TransposeNode>(node);
          auto shuffle = transpose_node->getShuffle();
          std::vector<int64_t> perm(shuffle.size());
          int i = 0;
          for(auto s: shuffle)
            perm[i++] = (int64_t)s;
          metawarenn::Attribute attr_pads("perm", perm);
          node_attributes.emplace_back(attr_pads);
          auto input = transpose_node->getInput().getNode();
          /* Checks if transpose layer lies between reshape nodes in Reshape -> Transpose -> Reshape order. If not,
            ignore the Transpose node to maintain the NCHW order*/
          if(!(llvm::dyn_cast<ReshapeNode>(input) && node_list[node_index+1]->getKindName() == "Reshape")) {
            ignored_transpose_nodes.insert(node_name);
            std::cout << "\nignored_transpose_node: " << node_name;
          }
          transpose_cnt++;
          break;
        }
        case Kinded::Kind::ReshapeNodeKind:
        {
          node_op_type = "Reshape";
          auto *reshape_node = llvm::cast<ReshapeNode>(node);
          auto input_name = reshape_node->getInput().generateNodeOutputName(true);
          std::string initializer_name = std::string(node_name + "_shape");
          auto dims = reshape_node->getDims();
          std::vector<int64_t> dims_vec(dims.size());
          std::vector<int> dims_;
          dims_.push_back(dims.size());
          int i = 0;
          for(auto dim: dims){
            dims_vec[i++] = dim;
          }
          metawarenn::Tensor reshape_tensor(initializer_name, dims_, get_mwnn_type_glow(ElemKind::Int64ITy), dims_vec);
          graph_->set_graph_initializers(reshape_tensor);
          graph_->initializer_names.insert(initializer_name);
          node_inputs.emplace_back(initializer_name);
          break;
        }
        case Kinded::Kind::LocalResponseNormalizationNodeKind:
        {
          node_op_type = "LRN";
          auto *lrn_node = llvm::cast<LocalResponseNormalizationNode>(node);
          metawarenn::Attribute attr_alpha("alpha", float(lrn_node->getAlpha()));
          node_attributes.emplace_back(attr_alpha);
          metawarenn::Attribute attr_beta("beta", float(lrn_node->getBeta()));
          node_attributes.emplace_back(attr_beta);
          metawarenn::Attribute attr_size("size", int64_t(2 * lrn_node->getHalfWindowSize() + 1));
          node_attributes.emplace_back(attr_size);
          metawarenn::Attribute attr_bias("bias", float(lrn_node->getK()));
          node_attributes.emplace_back(attr_bias);
          break;
        }
        case Kinded::Kind::MaxPoolNodeKind:
        {
          node_op_type = "MaxPool";
          auto *maxpool_node = llvm::cast<MaxPoolNode>(node);
          auto kernels = maxpool_node->getKernels();
          auto strides = maxpool_node->getStrides();
          auto pads = maxpool_node->getPads();
          metawarenn::Attribute attr_dilations("dilations", std::vector<int64_t>{1,1});
          node_attributes.emplace_back(attr_dilations);
          metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>{int64_t(kernels[0]), int64_t(kernels[1])});
          node_attributes.emplace_back(attr_kernel_shape);
          metawarenn::Attribute attr_pad("pads", std::vector<int64_t>{int64_t(pads[0]), int64_t(pads[1]), int64_t(pads[2]), int64_t(pads[3])});
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int64_t>{int64_t(strides[0]), int64_t(strides[1])});
          node_attributes.emplace_back(attr_stride);
          // Remove the additional(second) output name from GLOW Function. Consider only one output for MaxPool Node.
          if(node_outputs.size() > 1)
            node_outputs.pop_back();
          break;
        }
        case Kinded::Kind::GemmNodeKind:
        {
          node_op_type = "Gemm";
          auto *gemm_node = llvm::cast<GemmNode>(node);
          metawarenn::Attribute attr_alpha("alpha", float(gemm_node->getAlpha()));
          node_attributes.emplace_back(attr_alpha);
          metawarenn::Attribute attr_beta("beta", float(gemm_node->getBeta()));
          node_attributes.emplace_back(attr_beta);
          metawarenn::Attribute attr_transA("transA", int64_t(gemm_node->getTransposeA()));
          node_attributes.emplace_back(attr_transA);
          metawarenn::Attribute attr_transB("transB", int64_t(gemm_node->getTransposeB()));
          node_attributes.emplace_back(attr_transB);
          break;
        }
        case Kinded::Kind::ConcatNodeKind:
        {
          node_op_type = "Concat";
          auto *concat_node = llvm::cast<ConcatNode>(node);
          metawarenn::Attribute attr_axis("axis", (int64_t)1);//int(inputs[0].dims().size())});
          node_attributes.emplace_back(attr_axis);
          break;
        }
        case Kinded::Kind::BatchNormalizationNodeKind:
        {
          node_op_type = "BatchNormalization";
          auto *batchnorm_node = llvm::cast<BatchNormalizationNode>(node);
          metawarenn::Attribute attr_epsilon("epsilon", batchnorm_node->getEpsilon());
          node_attributes.emplace_back(attr_epsilon);
          metawarenn::Attribute attr_momentum("momentum", batchnorm_node->getMomentum());
          node_attributes.emplace_back(attr_momentum);
          auto mean = batchnorm_node->getMean();
          auto var = batchnorm_node->getVar();
          auto scale = batchnorm_node->getScale();
          std::vector<int> mean_dims(mean.dims().size());
          std::vector<int> var_dims(var.dims().size());
          std::vector<int> scale_dims(scale.dims().size());
          break;
        }
        case Kinded::Kind::ClipNodeKind:
        {
          node_op_type = "Clip";
          auto *clip_node = llvm::cast<ClipNode>(node);
          metawarenn::Tensor min_tensor(node_name + "_min", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{(float)clip_node->getMin()});
          graph_->set_graph_initializers(min_tensor);
          graph_->initializer_names.insert(min_tensor.get_name());
          node_inputs.emplace_back(min_tensor.get_name());
          metawarenn::Tensor max_tensor(node_name + "_max", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{(float)clip_node->getMax()});
          graph_->set_graph_initializers(max_tensor);
          graph_->initializer_names.insert(max_tensor.get_name());
          node_inputs.emplace_back(max_tensor.get_name());
          break;
        }
        case Kinded::Kind::FullyConnectedNodeKind:
        {
          node_op_type = "Gemm";
          break;
        }
        case Kinded::Kind::SoftMaxNodeKind:
        {
          node_op_type = "Softmax";
          // Remove the additional(second) input name from GLOW Function. Consider only one input for Softmax Node.
          if(node_inputs.size() > 1)
            node_inputs.pop_back();
          break;
        }
        case Kinded::Kind::SliceNodeKind:
        {
          node_op_type = "Slice";
          auto *slice_node = llvm::cast<SliceNode>(node);
          auto input_dims = slice_node->getInput().dims();
          auto starts = slice_node->getStart();
          auto outs = slice_node->getResult().dims();
          if(starts.size() != outs.size()) {
            std::cout << "Mismatch with starts and result dimensions.";
            exit(1);
          }
          auto size = starts.size();
          std::vector<int32_t> starts_vec(size);
          std::vector<int32_t> ends_vec(size);

          for (unsigned b = 0, e = size; b < e; ++b) {
            starts_vec[b] = starts[b];
            ends_vec[b] = outs[b] + starts[b];
          }
          metawarenn::Tensor tensor_starts(node_name + "_starts", std::vector<int>{int(size)}, ElementType::element_type::int32_, starts_vec);
          graph_->set_graph_initializers(tensor_starts);
          graph_->initializer_names.insert(tensor_starts.get_name());
          node_inputs.emplace_back(tensor_starts.get_name());
          metawarenn::Tensor tensor_ends(node_name + "ends", std::vector<int>{int(size)}, ElementType::element_type::int32_, ends_vec);
          graph_->set_graph_initializers(tensor_ends);
          graph_->initializer_names.insert(tensor_ends.get_name());
          node_inputs.emplace_back(tensor_ends.get_name());
          break;
        }
        case Kinded::Kind::PowNodeKind:
        {
          node_op_type = "Pow";
          break;
        }
        case Kinded::Kind::TopKNodeKind:
        {
          node_op_type = "TopK";
          auto *topk_node = llvm::cast<TopKNode>(node);
          auto k_val = topk_node->getK();
          metawarenn::Tensor k_tensor("K", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{float(k_val)});
          graph_->set_graph_initializers(k_tensor);
          graph_->initializer_names.insert(k_tensor.get_name());
          break;
        }
        case Kinded::Kind::ArgMaxNodeKind:
        {
          node_op_type = "ArgMax";
          auto *argmax_node = llvm::cast<ArgMaxNode>(node);
          metawarenn::Attribute attr_axis("axis", int64_t(argmax_node->getAxis()));
          metawarenn::Attribute attr_keep_dims("keepDims", int64_t(argmax_node->getKeepDims()));
          node_attributes.emplace_back(attr_axis);
          break;
        }
        case Kinded::Kind::ArgMinNodeKind:
        {
          node_op_type = "ArgMin";
          auto *argmin_node = llvm::cast<ArgMinNode>(node);
          metawarenn::Attribute attr_axis("axis", int64_t(argmin_node->getAxis()));
          node_attributes.emplace_back(attr_axis);
          metawarenn::Attribute attr_keep_dims("keepDims", int64_t(argmin_node->getKeepDims()));
          node_attributes.emplace_back(attr_keep_dims);
          break;
        }
        case Kinded::Kind::PReluNodeKind:
        {
          node_op_type = "PRelu";
          auto *prelu_node = llvm::cast<PReluNode>(node);
          auto slope = prelu_node->getSlope();
          if (const auto *BN = llvm::dyn_cast<BroadcastNode>(slope)) {
            node_inputs.emplace_back(BN->getInput().getNode()->getName());
          }
          else if (auto *SN = llvm::dyn_cast<SplatNode>(slope)) {
            glow::Tensor scalar = {SN->getValue()};
            auto dims = scalar.dims();
            std::vector<int> dims_vec(dims.size());
            int i = 0;
            for(auto dim: dims)
              dims_vec[i++] = int(dim);
            std::vector<float> scalar_data(dims.size());
            auto handle = scalar.getHandle<float>();
            i = 0;
            for (auto elem : handle) {
              scalar_data[i++] = elem;
            }
            node_inputs.emplace_back(SN->getName());
            metawarenn::Tensor slope_tensor(SN->getName(), dims_vec, ElementType::element_type::float_, scalar_data);
            graph_->set_graph_initializers(slope_tensor);
            graph_->initializer_names.insert(slope_tensor.get_name());
          }
          break;
        }
        case Kinded::Kind::GatherNodeKind:
        {
          node_op_type = "Gather";
          auto *gather_node = llvm::cast<GatherNode>(node);
          auto batch_dims = gather_node->getBatchDims();
          metawarenn::Attribute attr_axis("axis", (int64_t)batch_dims);
          node_attributes.emplace_back(attr_axis);
          break;
        }
        case Kinded::Kind::MulNodeKind:
        {
          node_op_type = "Mul";
          break;
        }
        case Kinded::Kind::DivNodeKind:
        {
          node_op_type = "Div";
          break;
        }
        case Kinded::Kind::SubNodeKind:
        {
          node_op_type = "Sub";
          break;
        }
        case Kinded::Kind::AbsNodeKind:
        {
          node_op_type = "Abs";
          break;
        }
        case Kinded::Kind::AndNodeKind:
        {
          node_op_type = "And";
          break;
        }
        case Kinded::Kind::ExpNodeKind:
        {
          node_op_type = "Exp";
          break;
        }
        case Kinded::Kind::MaxNodeKind:
        {
          node_op_type = "Max";
          break;
        }
        case Kinded::Kind::MinNodeKind:
        {
          node_op_type = "Min";
          break;
        }
        case Kinded::Kind::PadNodeKind:
        {
          node_op_type = "Pad";
          auto *pad_node = llvm::cast<PadNode>(node);
          auto pads = pad_node->getPads();
          metawarenn::Tensor pads_tensor(node_name + "_pads", std::vector<int>{(int)pads.size()}, ElementType::element_type::int64_, std::vector<int64_t>{pads[0], pads[1], pads[2], pads[3]});
          node_inputs.emplace_back(pads_tensor.get_name());
          metawarenn::Tensor value_tensor(node_name + "_value", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{(float)pad_node->getValue()});
          node_inputs.emplace_back(value_tensor.get_name());
          metawarenn::Attribute attr_mode("mode", int64_t(pad_node->getMode()));
          node_attributes.emplace_back(attr_mode);
          break;
        }
        case Kinded::Kind::CeilNodeKind:
        {
          node_op_type = "Ceil";
          break;
        }
        case Kinded::Kind::FloorNodeKind:
        {
          node_op_type = "Floor";
          break;
        }
        case Kinded::Kind::SwishNodeKind:
        {
          node_op_type = "HardSwish";
          break;
        }
        case Kinded::Kind::LeakyReluNodeKind:
        {
          node_op_type = "LeakyRelu";
          auto *lrelu_node = llvm::cast<LeakyReluNode>(node);
          metawarenn::Attribute attr_alpha("alpha", (float)lrelu_node->getAlpha());
          node_attributes.emplace_back(attr_alpha);
          break;
        }
        case Kinded::Kind::LogNodeKind:
        {
          node_op_type = "Log";
          break;
        }
        case Kinded::Kind::MatMulNodeKind:
        {
          node_op_type = "MatMul";
          break;
        }
        case Kinded::Kind::NonMaxSuppressionNodeKind:
        {
          node_op_type = "NonMaxSuppression";
          auto *nms_node = llvm::cast<NonMaxSuppressionNode>(node);
          metawarenn::Attribute attr_cpb("center_point_box", (int64_t)nms_node->getCenterPointBox());
          node_attributes.emplace_back(attr_cpb);
          metawarenn::Tensor max_out_box_tensor(node_name + "_max_out_box", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{float(nms_node->getMaxOutputBoxesPerClass())});
          graph_->set_graph_initializers(max_out_box_tensor);
          graph_->initializer_names.insert(max_out_box_tensor.get_name());
          node_inputs.emplace_back(max_out_box_tensor.get_name());
          metawarenn::Tensor iou_thresh_tensor(node_name + "_iou_thresh", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{nms_node->getIouThreshold()});
          graph_->set_graph_initializers(iou_thresh_tensor);
          graph_->initializer_names.insert(iou_thresh_tensor.get_name());
          node_inputs.emplace_back(iou_thresh_tensor.get_name());
          metawarenn::Tensor score_threshold_tensor(node_name + "_score_thresh", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{nms_node->getScoreThreshold()});
          graph_->set_graph_initializers(score_threshold_tensor);
          graph_->initializer_names.insert(score_threshold_tensor.get_name());
          node_inputs.emplace_back(score_threshold_tensor.get_name());
          break;
        }
        case Kinded::Kind::NotNodeKind:
        {
          node_op_type = "Not";
          break;
        }
        case Kinded::Kind::BatchedReduceMeanNodeKind:
        {
          node_op_type = "ReduceMean";
          auto *reduce_mean_node = llvm::cast<BatchedReduceMeanNode>(node);
          auto axes = reduce_mean_node->getAxes();
          std::vector<int64_t> axes_vec(axes.size());
          int i = 0;
          for(auto ax: axes_vec)
            axes_vec[i++] = axes[i];
          metawarenn::Attribute attr_axes("axes", axes_vec);
          node_attributes.emplace_back(attr_axes);
          break;
        }
        case Kinded::Kind::BatchedReduceMinNodeKind:
        {
          node_op_type = "ReduceMin";
          auto *reduce_min_node = llvm::cast<BatchedReduceMinNode>(node);
          auto axes = reduce_min_node->getAxes();
          std::vector<int64_t> axes_vec(axes.size());
          int i = 0;
          for(auto ax: axes_vec)
            axes_vec[i++] = axes[i];
          metawarenn::Attribute attr_axes("axes", axes_vec);
          node_attributes.emplace_back(attr_axes);
          break;
        }
        case Kinded::Kind::BatchedReduceMaxNodeKind:
        {
          node_op_type = "ReduceMax";
          auto *reduce_max_node = llvm::cast<BatchedReduceMaxNode>(node);
          auto axes = reduce_max_node->getAxes();
          std::vector<int64_t> axes_vec(axes.size());
          int i = 0;
          for(auto ax: axes_vec)
            axes_vec[i++] = axes[i];
          metawarenn::Attribute attr_axes("axes", axes_vec);
          node_attributes.emplace_back(attr_axes);
          break;
        }
        case Kinded::Kind::BatchedReduceSumSquareNodeKind:
        case Kinded::Kind::BatchedReduceAddNodeKind:
        {
          node_op_type = "ReduceSum";
          auto *reduce_add_node = llvm::cast<BatchedReduceAddNode>(node);
          metawarenn::Attribute attr_axes("axes", (int64_t)reduce_add_node->getAxis());
          node_attributes.emplace_back(attr_axes);
          break;
        }
        case Kinded::Kind::ResizeNearestNodeKind:
        {
          node_op_type = "Resize";
          auto *resize_near_node = llvm::cast<ResizeNearestNode>(node);
          auto scales = resize_near_node->getScale();
          std::vector<float> scale_vec(scales.size());
          int i = 0;
          for(auto scale: scales)
            scale_vec[i++] = (float)scale;
          metawarenn::Tensor scales_tensor(node_name + "_scales", std::vector<int>{(int)scales.size()}, ElementType::element_type::float_, scale_vec);
          graph_->set_graph_initializers(scales_tensor);
          graph_->initializer_names.insert(scales_tensor.get_name());
          node_inputs.emplace_back(scales_tensor.get_name());
          metawarenn::Attribute attr_trans_mode("coordinate_transformation_mode", "asymmetric");
          node_attributes.emplace_back(attr_trans_mode);
          metawarenn::Attribute attr_mode("mode", "nearest");
          node_attributes.emplace_back(attr_mode);
          metawarenn::Attribute attr_near_mode("nearest_mode", "floor");
          node_attributes.emplace_back(attr_near_mode);
          break;
        }
        case Kinded::Kind::ResizeBilinearNodeKind:
        {
          node_op_type = "Resize";
          auto *resize_bilinear_node = llvm::cast<ResizeBilinearNode>(node);
          auto scales = resize_bilinear_node->getScale();
          std::vector<float> scale_vec(scales.size());
          int i = 0;
          for(auto scale: scales)
            scale_vec[i++] = (float)scale;
          metawarenn::Tensor scales_tensor(node_name + "_scales", std::vector<int>{(int)scales.size()}, ElementType::element_type::float_, scale_vec);
          graph_->set_graph_initializers(scales_tensor);
          graph_->initializer_names.insert(scales_tensor.get_name());
          node_inputs.emplace_back(scales_tensor.get_name());
          metawarenn::Attribute attr_trans_mode("coordinate_transformation_mode", "asymmetric");
          node_attributes.emplace_back(attr_trans_mode);
          metawarenn::Attribute attr_mode("mode", "linear");
          node_attributes.emplace_back(attr_mode);
          break;
        }
        case Kinded::Kind::ROIAlignNodeKind:
        {
          node_op_type = "RoiAlign";
          auto *roi_align_node = llvm::cast<ROIAlignNode>(node);
          switch(roi_align_node->getMode()) {
            case PoolingMode::AVG: {
              metawarenn::Attribute attr_mode("mode", "avg");
              node_attributes.emplace_back(attr_mode);
              break;
            }
            case PoolingMode::MAX: {
              metawarenn::Attribute attr_mode("mode", "max");
              node_attributes.emplace_back(attr_mode);
              break;
            }
          }
          metawarenn::Attribute attr_out_h("output_height", (int64_t)roi_align_node->getOutputHeight());
          node_attributes.emplace_back(attr_out_h);
          metawarenn::Attribute attr_out_w("output_width", (int64_t)roi_align_node->getOutputWidth());
          node_attributes.emplace_back(attr_out_w);
          metawarenn::Attribute attr_s_ratio("sampling_ratio", (int64_t)roi_align_node->getSamplingRatio());
          node_attributes.emplace_back(attr_s_ratio);
          metawarenn::Attribute attr_s_scale("spatial_scale", (float)roi_align_node->getSpatialScale());
          node_attributes.emplace_back(attr_s_scale);
          break;
        }
        case Kinded::Kind::RoundNodeKind:
        {
          node_op_type = "Round";
          break;
        }
        case Kinded::Kind::SigmoidNodeKind:
        {
          node_op_type = "Sigmoid";
          break;
        }
        case Kinded::Kind::SpaceToDepthNodeKind:
        {
          node_op_type = "SpaceToDepth";
          break;
        }
        case Kinded::Kind::SqrtNodeKind:
        {
          node_op_type = "Sqrt";
          break;
        }
        case Kinded::Kind::TanhNodeKind:
        {
          node_op_type = "Tanh";
          break;
        }
        case Kinded::Kind::TileNodeKind:
        {
          node_op_type = "Tile";
          auto *tile_node = llvm::cast<TileNode>(node);
          std::vector<std::pair<unsigned_t, unsigned_t>> info;
          std::vector<size_t> repeats;
          const TileNode *tile = tile_node;
          while (tile) {
            info.insert(info.begin(), {tile->getAxis(), tile->getCount()});
            if (const auto *TN = llvm::dyn_cast<TileNode>(tile->getInput().getNode())) {
              tile = TN;
            } else {
              break;
            }
          }
          if (&repeats) {
            unsigned_t numDims = tile->getInput().dims().size();
            auto aB = info.begin();
            for (unsigned_t i = 0; i < numDims; ++i, ++aB) {
              if (aB == info.end() || aB->first != i) {
                aB = info.insert(aB, {i, 1});
              }
            }
            for (size_t b = 0, e = info.size(); b < e; ++b) {
              repeats.push_back(info[b].second);
            }
          }
          auto repeats_arr = llvm::makeArrayRef(repeats);
          std::vector<float> repeats_vec(repeats_arr.size());
          int i = 0;
          for(auto rep: repeats_arr)
            repeats_vec[i++] = (float)rep;
          metawarenn::Tensor repeats_tensor(node_name + "_repeats", std::vector<int>{(int)repeats_arr.size()}, ElementType::element_type::float_, repeats_vec);
          graph_->set_graph_initializers(repeats_tensor);
          graph_->initializer_names.insert(repeats_tensor.get_name());
          break;
        }
        case Kinded::Kind::ConvTransposeNodeKind:
        {
          node_op_type = "ConvTranspose";
          auto *conv_trans_node = llvm::cast<ConvTransposeNode>(node);
          auto kernels = conv_trans_node->getKernels();
          auto dilations = conv_trans_node->getDilation();
          auto strides = conv_trans_node->getStrides();
          auto pads = conv_trans_node->getPads();
          auto group = conv_trans_node->getGroup();
          metawarenn::Attribute attr_dilate("dilations", std::vector<int64_t>{int64_t(dilations[0]), int64_t(dilations[1])});
          node_attributes.emplace_back(attr_dilate);
          metawarenn::Attribute attr_group("group", int64_t(group));
          node_attributes.emplace_back(attr_group);
          metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>{(int64_t)kernels[0], (int64_t)kernels[1]});
          node_attributes.emplace_back(attr_kernel_shape);
          auto input_dims = conv_trans_node->getInput().dims();
          PaddingTLBR pdim(pads);
          ShapeHW kdim(kernels);
          ShapeHW sdim(strides);
          int64_t depth = (int64_t)input_dims[0] * (int64_t)group;
          int64_t outsx = (input_dims[1] - 1) * sdim.height + (kdim.height - 1) * dilations[0] + 1 -
                        pdim.top - pdim.bottom;
          int64_t outsy = (input_dims[2] - 1) * sdim.width + (kdim.width - 1) * dilations[1] + 1 -
                        pdim.left - pdim.right;
          std::vector<int64_t> output_shape = {(int64_t)input_dims[0], outsx, outsy, depth};
          metawarenn::Attribute attr_output_shape("output_shape", output_shape);
          node_attributes.emplace_back(attr_kernel_shape);
          metawarenn::Attribute attr_pad("pads", std::vector<int64_t>{int64_t(pads[0]), int64_t(pads[1]), int64_t(pads[2]), int64_t(pads[3])});
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int64_t>{int64_t(strides[0]), int64_t(strides[1])});
          node_attributes.emplace_back(attr_stride);
          break;
        }
        case Kinded::Kind::QuantizeNodeKind:
        {
          node_op_type = "QuantizeLinear";
          auto *quant_node = llvm::cast<QuantizeNode>(node);
          std::string name = quant_node->getNthResult(0).getNode()->getName();
          CreateMWNNQuantParams(quant_node->getNthResult(0), name);
          node_inputs.emplace_back(name + "_scale");
          node_inputs.emplace_back(name + "_zero_point");
          break;
        }
        case Kinded::Kind::DequantizeNodeKind:
        {
          node_op_type = "DequantizeLinear";
          auto *dequant_node = llvm::cast<DequantizeNode>(node);
          std::string name = dequant_node->getNthInput(0).getNode()->getName();
          CreateMWNNQuantParams(dequant_node->getNthResult(0), name);
          node_inputs.emplace_back(name + "_scale");
          node_inputs.emplace_back(name + "_zero_point");
          break;
        }
        case Kinded::Kind::RescaleQuantizedNodeKind:
        {
          auto *rq_node = llvm::cast<RescaleQuantizedNode>(node);
          auto input = rq_node->getNthInput(0);
          std::string name = rq_node->getName();

          auto itr = quant_op_mapper.find(name);
          auto rq_output = node_outputs[0];
          if(quantize_encounter && itr == quant_op_mapper.end()) {
            CreateMWNNQuantParams(input, name);
            CreateQDQNodes(node_inputs[0], node_outputs[0], node_name);
          }
          continue;
          break;
        }
        default:
          break;
        }
        // Check to avoid empty node creation for removed nodes
        if(!onnx_unsupported_nodes.count(node->getKind())) {
          metawarenn::Node m_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
          graph_->set_graph_nodes(m_node);
          // keeps track of CHW format inputs & initializer before the first transpose node
          if(transpose_cnt == 0) {
            for(auto inp : node_inputs)
              node_inputs_map.insert(inp);
          }
          if(global_input_name == "")
            global_input_name = node_inputs.front();
          global_output_name = node_outputs.back();
          std::cout << "\nglobal_input_name: " << global_input_name;
        }
      }
      node_index++;
    }
    optimizer::PassManager manager;
    if(CHW_TO_HWC)
    {
      for (auto g_t : graph_->get_graph_ip_tensor()) {
        if(g_t.get_dims().size() == 4) {
          /*std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";*/
          optimizer::ConvertLayout cl(graph_, g_t, CHW_TO_HWC, 0, 0, false);
          manager.register_pass(cl);
        }
      }
    }
    std::cout << "\nremove_transpose : " << remove_transpose;
    if(HWC_TO_CHW)
    {
      for (auto g_t : graph_->get_graph_initializers()) {
        auto dims = g_t.get_dims();
        if(g_t.get_dims().size() == 4) {
          if(remove_transpose && node_inputs_map.count(g_t.get_name()))
            continue;
          /*std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";*/
          ::metawarenn::optimizer::ConvertLayout cl(graph_, g_t, 0, HWC_TO_CHW, 0, true);
          manager.register_pass(cl);
        }
      }
      //Subgraph from other backends is already in CHW order
      /*if(graph_count == 1) {
        for (auto g_t : graph_->get_graph_ip_tensor()) {
          if(g_t.get_dims().size() == 4) {
            /*std::cout << "\n Name : " << g_t.get_name();
            std::cout << "\t Dims : ";
            for (auto dim : g_t.get_dims())
              std::cout << dim << ",";*/
            /*::metawarenn::optimizer::ConvertLayout cl(graph_, g_t, 0, HWC_TO_CHW, 0, false);
            manager.register_pass(cl);
          }
        }
      }*/
    }
    auto m_nodes = graph_->get_graph_nodes();
    int transpose_removal = 0;
    for (int node_idx = 0; node_idx < graph_->get_graph_nodes().size(); node_idx++) {
      auto g_n = m_nodes[node_idx];
      /*if(g_n.get_op_type() == "Relu") {
        optimizer::FuseRelu fr(graph_, g_n);
        //std::cout << "\n MetaWareNNCC : " << fr.get_name();
        manager.register_pass(fr);
      }
      else*/ if((g_n.get_op_type() == "Transpose")) {
        if(remove_transpose && (transpose_removal == 0)) {
          optimizer::RemoveTranspose rt(graph_, g_n);
          //std::cout << "\n MetaWareNNCC : " << rt.get_name();
          manager.register_pass(rt);
          transpose_removal++;
        }
        else if(g_n.get_inputs()[0] == graph_->get_graph_ip_names()[0])
        {
          optimizer::RemoveTranspose rt(graph_, g_n);
          //std::cout << "\n MetaWareNNCC : " << rt.get_name();
          manager.register_pass(rt);
          transpose_removal++;
        }
        else if (ignored_transpose_nodes.count(g_n.get_name()))
        {
          optimizer::RemoveTranspose rt(graph_, g_n);
          //std::cout << "\n MetaWareNNCC : " << rt.get_name();
          manager.register_pass(rt);
        }
      }
      else if(g_n.get_op_type() == "Reshape" && g_n.get_inputs()[0] == graph_->get_graph_ip_names()[0]) {
          optimizer::RemoveReshape rt(graph_, g_n);
          //std::cout << "\n MetaWareNNCC : " << rt.get_name();
          manager.register_pass(rt);
        }
    }
    /*optimizer::CalculateOffset co(graph_);
    manager.register_pass(co);*/
    manager.run_passes();
    #if !EXECUTABLE_GRAPH_SERIALIZATION
    write_onnx_proto(graph_);
    #endif

    auto graph_ip_names = graph_->get_graph_ip_names();
    for (auto g_n : graph_->get_graph_nodes()) {
      for (auto n_ip : g_n.get_inputs()) {
        if(!(graph_->initializer_names.count(n_ip)) && !(std::count(graph_ip_names.begin(), graph_ip_names.end(), n_ip))) {
          if (graph_->get_node_producers().count(n_ip)) {
            graph_->set_node_consumer(n_ip, g_n.get_name());
          }
        }
      }
      for (auto n_op : g_n.get_outputs()) {
        graph_->set_node_producer(n_op, g_n.get_name());
      }
    }
    for (auto itr : graph_->get_node_producers()) {
      std::cout << "\n Produced Tensor : " << itr.first;
      std::cout << "\n      Producer Node : " << itr.second;
    }
    for (auto itr : graph_->get_node_consumers()) {
      std::cout << "\n Consumed Tensor : " << itr.first;
      auto& vitr = itr.second;
      for (auto node_name : vitr) {
          std::cout << "\n      Consumer Node - " << node_name;
      }
    }
    #if EXECUTABLE_GRAPH_SERIALIZATION
    exe_graph_ = std::make_shared<metawarenn::ExecutableGraph>(*graph_);
    #endif

    #if INVOKE_NNAC
      std::cout << "\n ---------------------------Graph----------------------------- \n";
      std::cout << "\n Graph Name : " << graph_->get_name();
    ::MWNN::MWNNGraphProto graph_proto;
    graph_proto.set_name(graph_->get_name());
    for (auto g_ip : graph_->get_graph_ip_names())
      graph_proto.add_ip_name((g_ip));
    for (auto g_op : graph_->get_graph_op_names())
      graph_proto.add_op_name((g_op));

    std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
    for (auto g_ip : graph_->get_graph_ip_tensor()) {
      std::cout << "\n Input Name : " << g_ip.get_name();
      std::cout << "\n Data Type : " << g_ip.get_type();
      std::cout << "\n Input Dims : ";
      auto input = graph_proto.add_input();
      input->set_name(g_ip.get_name());
      input->set_type(g_ip.get_type());
      for (auto dim : g_ip.get_dims()) {
        std::cout << dim << ",";
        input->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
    for (auto g_op : graph_->get_graph_op_tensor()) {
      std::cout << "\n Output Name : " << g_op.get_name();
      std::cout << "\n Data Type : " << g_op.get_type();
      std::cout << "\n Output Dims : ";
      auto output = graph_proto.add_output();
      output->set_name(g_op.get_name());
      output->set_type(g_op.get_type());
      for (auto dim : g_op.get_dims()) {
        std::cout << dim << ",";
        output->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Nodes-------------------------- \n";
    for (auto g_n : graph_->get_graph_nodes()) {
      std::cout << "\n ================================================================ \n";
      std::cout << "\n Node Name : " << g_n.get_name();
      std::cout << "\n Op Type : " << g_n.get_op_type();
      auto node = graph_proto.add_node();
      node->set_name(g_n.get_name());
      auto op_type = g_n.get_op_type();
      node->set_op_type(op_type == "DepthwiseConv" ? "Conv" : op_type);
      for (auto n_ip : g_n.get_inputs()) {
        std::cout << "\n Input : n_ip : " << n_ip;
        node->add_ip_name((n_ip));
      }
      for (auto n_op : g_n.get_outputs()) {
        std::cout << "\n Output : n_op : " << n_op;
        node->add_op_name((n_op));
      }
      std::cout << "\n ---------------------------------------------------------------- ";
      for (auto attribute : g_n.get_attributes()) {
        std::cout << "\n Attribute Name : " << attribute.get_name();
        std::cout << "\n Attribute Data : ";
        auto attr = node->add_attribute();
        attr->set_name(attribute.get_name());
        attr->set_type(attribute.get_type());
        if(attribute.get_type() == 6) { //int data
          for(int i = 0; i < attribute.get_int_data().size(); i++){
            attr->add_int_data(attribute.get_int_data()[i]);
            std::cout << attribute.get_int_data()[i] << ",";
          }
        }
        else if(attribute.get_type() == 3) { //float data
          for(int i = 0; i < attribute.get_float_data().size(); i++){
            attr->add_float_data(attribute.get_float_data()[i]);
            std::cout << attribute.get_float_data()[i] << ",";
          }
        }
        else if(attribute.get_type() == 12) { //string data
          for(int i = 0; i < attribute.get_string_data().size(); i++){
            attr->add_string_data(attribute.get_string_data()[i]);
            std::cout << attribute.get_string_data()[i] << ",";
          }
        }
      }
    }
    std::cout << "\n -----------------------Graph Tensors-------------------------- \n";
    for (auto g_t : graph_->get_graph_initializers()) {
      auto initializer = graph_proto.add_initializer();
      initializer->set_name(g_t.get_name());
      initializer->set_type(g_t.get_type());
      std::cout << "\n Name : " << g_t.get_name();
      std::cout << "\n Type : " << g_t.get_type();
      std::cout << "\n Dims : ";
      for (auto dim : g_t.get_dims()) {
        std::cout << dim << ",";
        initializer->add_dims(dim);
      }
      //std::cout << "\n Tensor values : ";
      for (auto t_val : g_t.get_tensor()) {
        //std::cout << t_val << ",";
        initializer->add_float_data(t_val);
      }
    }
    std::cout << "\n -----------------------Graph Tensor Producers-------------------------- \n";
    for (auto producer : graph_->get_node_producers()) {
      std::cout << "\n Produced Tensor : " << producer.first;
      std::cout << "\n      Producer Node : " << producer.second;
      auto pnode = graph_proto.add_producers();
      pnode->set_tensor_name(producer.first);
      pnode->add_node_name(producer.second);
    }
    std::cout << "\n -----------------------Graph Tensor Consumers-------------------------- \n";
    for (auto consumer : graph_->get_node_consumers()) {
      std::cout << "\n Consumed Tensor : " << consumer.first;
      auto& consumer_nodes = consumer.second;
      auto cnode = graph_proto.add_consumers();
      cnode->set_tensor_name(consumer.first);
      for (auto node_name : consumer_nodes) {
        std::cout << "\n      Consumer Node - " << node_name;
        cnode->add_node_name(node_name);
        }
    }
    std::string g_name = graph_->get_name();
    char* op_path = nullptr;
    op_path = getenv("NNAC_DUMPS_PATH");
    if(!IsPathExist(std::string(op_path))) {
      int check = mkdir(op_path, 0777);
      if(check != 0) {
        std::cout << "\nPlease check the directory path to store the serialized binary!!!!!";
        exit(1);
      }
    }
    auto proto_bin = std::string(op_path) + std::string(g_name) + ".bin";

    int fp = open(proto_bin.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::cout << fp;
    std::cout << graph_proto.SerializeToFileDescriptor(fp);
    close(fp);

    char* lib_path = nullptr;
    lib_path = getenv("METAWARENN_LIB_PATH");
    if(!IsPathExist(std::string(lib_path)))
      std::cout << "\nPlease check the MetaWareNN Library path!!!";
    std::cout << "\n\n=================Initiating NNAC python script via shell script======================\n";
    std::string cmd = "bash " + std::string(lib_path) +"/mwnnconvert/mwnn_convert.sh " + proto_bin + " " + op_path + " " + g_name + " " + std::to_string(graph_count);
    const char *command = cmd.c_str();
    system(command);
    #endif
}
void MetaWareNNFunction::findIOPlaceholders(Function *F) {
  for (auto const &V : F->getParent()->getPlaceholders()) {
    if (!usedInFunction(V, F)) {
      continue;
    }
    if (getOutputSave(F, V)) {
      //std::cout << "\n V->getName() in outputs_: " << std::string(V->getName());
      outputs_.push_back(V);
    } else {
      //std::cout << "\n V->getName() in inputs_: " << std::string(V->getName());
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
    graph_outputs[graph_->get_graph_op_names()[0]] = (float*)tensor->getUnsafePtr();

  }

  // **************************************** Calls to invoke the MetaWareNN Inference API ************************************
  #if EXECUTABLE_GRAPH_SERIALIZATION
  InferenceApi mwapi;

  std::vector<std::string> ip_names = graph_->get_graph_ip_names();
  auto ip_shape = graph_->get_graph_ip_tensor()[0].get_dims();

  mwapi.prepareInput(graph_inputs[ip_names[0]], ip_shape);

  std::vector<std::string> op_names = graph_->get_graph_op_names();
  auto op_shape = graph_->get_graph_op_tensor()[0].get_dims();

  mwapi.prepareOutput(op_shape);

  mwapi.prepareGraph(graph_->get_name());

  mwapi.runGraph();

  mwapi.getOutput(graph_outputs[op_names[0]], op_shape);
  #endif

  // ******************************************* Call to invoke the local run function *****************************************

  //convert_to_mwnn_format(*graph_, graph_inputs, graph_outputs, CHW_TO_HWC);
  return Error::success();
}
} // namespace metawarenn
