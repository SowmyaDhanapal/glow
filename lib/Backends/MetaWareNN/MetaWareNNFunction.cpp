#include "MetaWareNNFunction.h"

namespace metawarenn {


MetaWareNNFunction::MetaWareNNFunction(runtime::RuntimeBundle &&bundle, Function *F)
    : CompiledFunction(std::move(bundle)) {
    findIOPlaceholders(F);
    graph_count++;
    std::string subgraph_name = "MetaWareNN_" + std::to_string(graph_count);

    /*Create MetaWareNN High Level Graph Representation from Glow SubGraph Function*/
    mwnn_graph_ = std::make_shared<MWNNGraph>();
    mwnn_graph_->set_name(subgraph_name);

    std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
    std::cout << "\n MWNN Graph Name : " << mwnn_graph_->get_name();

    GraphPostOrderVisitor visitor(*F);
    auto node_list = visitor.getPostOrder();
    auto global_output_name = "";

    for (auto *node : node_list) {
        LOG(INFO) << "==============================================================================================================";
        std::string node_name;
        std::string node_op_type;
        std::vector<std::string> node_inputs;
        std::vector<std::string> node_outputs;
        std::vector<::metawarenn::MWNNAttribute> node_attributes;
        node_name = std::string(node->getName());
        auto kind = node->getKindName();
        LOG(INFO) << "Node Name: " << node_name << "\tOp type: " << node->getKindName();
        std::vector<std::string> non_op_types;
        non_op_types = {"Constant", "Placeholder", "Save"};
        if((std::count(non_op_types.begin(), non_op_types.end(), kind)) < 1)
        {
        switch (node->getKind())
        {
          case Kinded::Kind::ConvolutionNodeKind:
          {
            node_op_type = "Conv";
            auto *conv_node = llvm::cast<ConvolutionNode>(node);
            auto input_node = conv_node->getInput();
            auto input_name = input_node.generateNodeOutputName(true);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto filter_node_value = conv_node->getFilter();
            auto filter_name = filter_node_value.generateNodeOutputName(true);
            LOG(INFO) << "filter_name: " << filter_name;
            node_inputs.emplace_back(filter_name);
            mwnn_graph_->mwnn_initializer_names.insert(filter_name);
            auto *filter_constant = llvm::dyn_cast<glow::Constant>(filter_node_value.getNode());
            glow::Tensor filter_tensor = filter_constant->getPayload().clone();
            auto type = filter_tensor.getType();
            glow::ElemKind data_type = type.getElementType();
            ShapeNHWC filterDims(filter_node_value.dims());
            size_t wt_size = filterDims.n * filterDims.h * filterDims.w * filterDims.c;
            std::vector<float> weights(wt_size);
            std::vector<int> weight_dims(4);
            weight_dims[0] = filterDims.n;
            weight_dims[1] = filterDims.h;
            weight_dims[2] = filterDims.w;
            weight_dims[3] = filterDims.c;
            auto handle = filter_tensor.getHandle<float>();
            int i = 0;
            for (auto elem : handle) {
              weights[i++] = elem;
            }
            metawarenn::MWNNTensor mwnn_weight_tensor(filter_name, weight_dims, get_mwnn_type_glow(data_type), weights);
            mwnn_graph_->set_graph_initializers(mwnn_weight_tensor);
            auto bias_node_value = conv_node->getBias();
            auto bias_name = bias_node_value.generateNodeOutputName(true);
            // Check to avoid redundant constants in mwnn initializers
            if(!mwnn_graph_->mwnn_initializer_names.count(bias_name))
            {
              LOG(INFO) << "bias_name: " << bias_name;
              auto *bias_constant = llvm::dyn_cast<glow::Constant>(bias_node_value.getNode());
              glow::Tensor bias_tensor = bias_constant->getPayload().clone();
              auto handle1 = bias_tensor.getHandle<float>();
              auto base1 = handle1.getElementPtr({static_cast<unsigned long>(0)});
              std::vector<float> bias(filterDims.n);
              std::vector<int> bias_dims(1);
              bias_dims[0] = filterDims.n;
              i = 0;
              for (auto elem : handle1) {
                bias[i++] = elem;
              }
              node_inputs.emplace_back(bias_name);
              mwnn_graph_->mwnn_initializer_names.insert(bias_name);
              type = bias_tensor.getType();
              data_type = type.getElementType();
              metawarenn::MWNNTensor mwnn_bias_tensor(bias_name, bias_dims, get_mwnn_type_glow(data_type), bias);
              mwnn_graph_->set_graph_initializers(mwnn_bias_tensor);
            }
            auto dilations = conv_node->getDilation();
            auto strides = conv_node->getStrides();
            auto pads = conv_node->getPads();
            auto group = conv_node->getGroup();
            metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", std::vector<int>{int(dilations[0]), int(dilations[1])});
            node_attributes.emplace_back(mwnn_attr_dilate);
            metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
            node_attributes.emplace_back(mwnn_attr_stride);
            metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
            node_attributes.emplace_back(mwnn_attr_pad);
            metawarenn::MWNNAttribute mwnn_attr_group("group", std::vector<int>{int(group)});
            node_attributes.emplace_back(mwnn_attr_group);
            metawarenn::MWNNAttribute mwnn_attribute("activation", std::vector<int>{0});
            node_attributes.emplace_back(mwnn_attribute);
            metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{(int)filterDims.h, (int)filterDims.w});
            node_attributes.emplace_back(mwnn_attr_kernel_shape);
            auto output_name = conv_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
          }
          case Kinded::Kind::ReluNodeKind:
          {
            node_op_type = "Relu";
            auto *relu_node = llvm::cast<ReluNode>(node);
            auto input_name = relu_node->getInput().generateNodeOutputName(true);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = relu_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
          }
        case Kinded::Kind::AvgPoolNodeKind:
        {
          node_op_type = "GlobalAveragePool";
          auto *avgpool_node = llvm::cast<AvgPoolNode>(node);
          auto kernels = avgpool_node->getKernels();
          auto strides = avgpool_node->getStrides();
          auto pads = avgpool_node->getPads();
          metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{int(kernels[0]), int(kernels[1])});
          node_attributes.emplace_back(mwnn_attr_kernel_shape);
          metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
          node_attributes.emplace_back(mwnn_attr_stride);
          metawarenn::MWNNAttribute mwnn_attr_pads("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
          node_attributes.emplace_back(mwnn_attr_pads);
          auto input_name = avgpool_node->getInput().generateNodeOutputName(true);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = avgpool_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::AddNodeKind:
        {
          node_op_type = "Add";
          auto *add_node = llvm::cast<AddNode>(node);
          auto input1 = add_node->getLHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 1: " << input1;
          auto input2 = add_node->getRHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 2: " << input2;
          node_inputs.emplace_back(input1);
          node_inputs.emplace_back(input2);
          auto output_name = add_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::TransposeNodeKind:
        {
          node_op_type = "Transpose";
          auto *transpose_node = llvm::cast<TransposeNode>(node);
          auto input_name = transpose_node->getInput().generateNodeOutputName(true);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = transpose_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ReshapeNodeKind:
        {
          node_op_type = "Reshape";
          auto *reshape_node = llvm::cast<ReshapeNode>(node);
          auto input_name = reshape_node->getInput().generateNodeOutputName(true);
          std::string initializer_name = std::string(node_name + "shape");
          auto dims = reshape_node->getDims();
          std::vector<float> dims_vec(dims.size());
          std::vector<int> dims_;
          dims_.push_back(dims.size());
          int i = 0;
          for(auto dim: dims){
            dims_vec[i++] = dim;
          }
          metawarenn::MWNNTensor mwnn_reshape_tensor(initializer_name, dims_, get_mwnn_type_glow(ElemKind::Int64ITy), dims_vec);
          mwnn_graph_->set_graph_initializers(mwnn_reshape_tensor);
          node_inputs.emplace_back(input_name);
          node_inputs.emplace_back(initializer_name);
          mwnn_graph_->mwnn_initializer_names.insert(initializer_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = reshape_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::LocalResponseNormalizationNodeKind:
        {
          node_op_type = "LRN";
          auto *lrn_node = llvm::cast<LocalResponseNormalizationNode>(node);
          metawarenn::MWNNAttribute mwnn_attr_alpha("alpha", std::vector<int>{int(lrn_node->getAlpha())});
          node_attributes.emplace_back(mwnn_attr_alpha);
          metawarenn::MWNNAttribute mwnn_attr_beta("beta", std::vector<int>{int(lrn_node->getBeta())});
          node_attributes.emplace_back(mwnn_attr_beta);
          metawarenn::MWNNAttribute mwnn_attr_half_window_size("half_window_size", std::vector<int>{int(lrn_node->getHalfWindowSize())});
          node_attributes.emplace_back(mwnn_attr_half_window_size);
          auto input_name = lrn_node->getInput().generateNodeOutputName(true);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = lrn_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::MaxPoolNodeKind:
        {
          node_op_type = "MaxPool";
          auto *maxpool_node = llvm::cast<MaxPoolNode>(node);
          auto kernels = maxpool_node->getKernels();
          auto strides = maxpool_node->getStrides();
          auto pads = maxpool_node->getPads();
          metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{int(kernels[0]), int(kernels[1])});
          node_attributes.emplace_back(mwnn_attr_kernel_shape);
          metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
          node_attributes.emplace_back(mwnn_attr_stride);
          metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
          node_attributes.emplace_back(mwnn_attr_pad);
          auto input_name = maxpool_node->getInput().generateNodeOutputName(true);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = maxpool_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::GemmNodeKind:
        {
          node_op_type = "Gemm";
          auto *gemm_node = llvm::cast<GemmNode>(node);
          std::cout << "\n gemm inputs: " << gemm_node->getNumInputs();
          auto filter_node_value = gemm_node->getNthInput(1);
          auto filter_name = filter_node_value.generateNodeOutputName(true);
          mwnn_graph_->mwnn_initializer_names.insert(filter_name);
          node_inputs.emplace_back(filter_name);
          auto *filter_constant = llvm::dyn_cast<glow::Constant>(filter_node_value.getNode());
          glow::Tensor filter_tensor = filter_constant->getPayload().clone();
          auto type = filter_tensor.getType();
          glow::ElemKind data_type = type.getElementType();
          ShapeNHWC filterDims(filter_node_value.dims());
          size_t wt_size = filterDims.n * filterDims.h; //n - height, h - width
          std::vector<float> weights(wt_size);
          std::vector<int> weight_dims(filter_constant->dims().vec().size());
          weight_dims[0] = filterDims.n;
          weight_dims[1] = filterDims.h;
          auto handle = filter_tensor.getHandle<float>();
          int i = 0;
          for (auto elem : handle)
          {
              weights[i++] = elem;
          }
          metawarenn::MWNNTensor mwnn_weight_tensor(filter_name, weight_dims, get_mwnn_type_glow(data_type), weights);
          mwnn_graph_->set_graph_initializers(mwnn_weight_tensor);
          auto bias_node_value = gemm_node->getNthInput(2);
          auto bias_name = bias_node_value.generateNodeOutputName(true);
          LOG(INFO) << "bias_name: " << bias_name;
          node_inputs.emplace_back(bias_name);
          mwnn_graph_->mwnn_initializer_names.insert(bias_name);
          auto *bias_constant = llvm::dyn_cast<glow::Constant>(bias_node_value.getNode());
          glow::Tensor bias_tensor = bias_constant->getPayload().clone();
          auto handle1 = bias_tensor.getHandle<float>();
          auto base1 = handle1.getElementPtr({static_cast<unsigned long>(0)});
          std::vector<float> bias(filterDims.n);
          std::vector<int> bias_dims(1);
          bias_dims[0] = filterDims.n;
          i = 0;
          for (auto elem : handle1)
          {
              bias[i++] = elem;
          }
          type = bias_tensor.getType();
          data_type = type.getElementType();
          metawarenn::MWNNTensor mwnn_bias_tensor(bias_name, bias_dims, get_mwnn_type_glow(data_type), bias);
          mwnn_graph_->set_graph_initializers(mwnn_bias_tensor);
          metawarenn::MWNNAttribute mwnn_attr_alpha("alpha", std::vector<int>{int(gemm_node->getAlpha())});
          node_attributes.emplace_back(mwnn_attr_alpha);
          metawarenn::MWNNAttribute mwnn_attr_beta("beta", std::vector<int>{int(gemm_node->getBeta())});
          node_attributes.emplace_back(mwnn_attr_beta);
          auto input_name = gemm_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = gemm_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ConcatNodeKind:
        {
          node_op_type = "Concat";
          auto *concat_node = llvm::cast<ConcatNode>(node);
          for(int i = 0; i < concat_node->getInputs().size(); i++)
          {
            auto input_name = concat_node->getInputName(i);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
          }
          auto output_name = concat_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::BatchNormalizationNodeKind:
        {
          node_op_type = "BatchNormalization";
          auto *batchnorm_node = llvm::cast<BatchNormalizationNode>(node);
          batchnorm_node->getEpsilon();
          batchnorm_node->getMomentum();
          auto bias_node_value = batchnorm_node->getBias();
          auto bias_name = bias_node_value.generateNodeOutputName(true);
          LOG(INFO) << "bias_name: " << bias_name;
          node_inputs.emplace_back(bias_name);
          mwnn_graph_->mwnn_initializer_names.insert(bias_name);
          auto *bias_constant = llvm::dyn_cast<glow::Constant>(bias_node_value.getNode());
          glow::Tensor bias_tensor = bias_constant->getPayload().clone();
          auto handle1 = bias_tensor.getHandle<float>();
          auto base1 = handle1.getElementPtr({static_cast<unsigned long>(0)});
          std::vector<float> bias(bias_tensor.size());
          std::vector<int> bias_dims(1);
          bias_dims[0] = bias_tensor.dims()[0];
          int i = 0;
          for (auto elem : handle1)
          {
              bias[i++] = elem;
          }
          auto type = bias_tensor.getType();
          auto data_type = type.getElementType();
          metawarenn::MWNNTensor mwnn_bias_tensor(bias_name, bias_dims, get_mwnn_type_glow(data_type), bias);
          mwnn_graph_->set_graph_initializers(mwnn_bias_tensor);
          metawarenn::MWNNAttribute mwnn_attr_momentum("momentum", std::vector<float>{batchnorm_node->getMomentum()});
          node_attributes.emplace_back(mwnn_attr_momentum);
          metawarenn::MWNNAttribute mwnn_attr_epsilon("epsilon", std::vector<float>{batchnorm_node->getEpsilon()});
          node_attributes.emplace_back(mwnn_attr_epsilon);
          auto input_name = batchnorm_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          auto output_name = batchnorm_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ChannelShuffleNodeKind:
        {
          node_op_type = "ChannelShuffle";
          auto *channel_shuffle_node = llvm::cast<ChannelShuffleNode>(node);
          metawarenn::MWNNAttribute mwnn_attr_group("group", std::vector<int>{int(channel_shuffle_node->getGroup())});
          node_attributes.emplace_back(mwnn_attr_group);
          metawarenn::MWNNAttribute mwnn_attr_kernel("kernel", std::vector<int>{int(channel_shuffle_node->getKernel())});
          node_attributes.emplace_back(mwnn_attr_kernel);
          auto input_name = channel_shuffle_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          auto output_name = channel_shuffle_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ClipNodeKind:
        {
          node_op_type = "Clip";
          auto *clip_node = llvm::cast<ClipNode>(node);
          clip_node->getMax();
          clip_node->getMin();
          metawarenn::MWNNAttribute mwnn_attr_max("max", std::vector<float>{(clip_node->getMax())});
          node_attributes.emplace_back(mwnn_attr_max);
          metawarenn::MWNNAttribute mwnn_attr_min("min", std::vector<float>{(clip_node->getMax())});
          node_attributes.emplace_back(mwnn_attr_min);
          auto input_name = clip_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          auto output_name = clip_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::FullyConnectedNodeKind:
        {
          node_op_type = "FullyConnected";
          auto *fc_node = llvm::cast<FullyConnectedNode>(node);
          auto filter_node_value = fc_node->getWeights();
          auto filter_name = filter_node_value.generateNodeOutputName(true);
          auto *filter_constant = llvm::dyn_cast<glow::Constant>(filter_node_value.getNode());
          glow::Tensor filter_tensor = filter_constant->getPayload().clone();
          auto type = filter_tensor.getType();
          glow::ElemKind data_type = type.getElementType();
          ShapeNHWC filterDims(filter_node_value.dims());
          size_t wt_size = filterDims.n * filterDims.h; //n - height, h - width
          std::vector<float> weights(wt_size);
          std::vector<int> weight_dims(filter_constant->dims().vec().size());
          weight_dims[0] = filterDims.n;
          weight_dims[1] = filterDims.h;
          auto handle = filter_tensor.getHandle<float>();
          int i = 0;
          for (auto elem : handle)
          {
              weights[i++] = elem;
          }
          mwnn_graph_->mwnn_initializer_names.insert(filter_name);
          node_inputs.emplace_back(filter_name);
          metawarenn::MWNNTensor mwnn_weight_tensor(filter_name, weight_dims, get_mwnn_type_glow(data_type), weights);
          mwnn_graph_->set_graph_initializers(mwnn_weight_tensor);
          auto bias_node_value = fc_node->getBias();;
          auto bias_name = bias_node_value.generateNodeOutputName(true);
          LOG(INFO) << "bias_name: " << bias_name;
          node_inputs.emplace_back(bias_name);
          mwnn_graph_->mwnn_initializer_names.insert(bias_name);
          auto *bias_constant = llvm::dyn_cast<glow::Constant>(bias_node_value.getNode());
          glow::Tensor bias_tensor = bias_constant->getPayload().clone();
          auto handle1 = bias_tensor.getHandle<float>();
          auto base1 = handle1.getElementPtr({static_cast<unsigned long>(0)});
          std::vector<float> bias(filterDims.n);
          std::vector<int> bias_dims(1);
          bias_dims[0] = filterDims.n;
          i = 0;
          for (auto elem : handle1)
          {
              bias[i++] = elem;
          }
          type = bias_tensor.getType();
          data_type = type.getElementType();
          metawarenn::MWNNTensor mwnn_bias_tensor(bias_name, bias_dims, get_mwnn_type_glow(data_type), bias);
          mwnn_graph_->set_graph_initializers(mwnn_bias_tensor);
          auto input_name = fc_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = fc_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::SoftMaxNodeKind:
        {
          node_op_type = "Softmax";
          auto *softmax_node = llvm::cast<SoftMaxNode>(node);
          auto input_name = softmax_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = softmax_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        default:
          break;
        }
        metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
        mwnn_graph_->set_graph_nodes(mwnn_node);
        auto op_node = mwnn_node.get_node();
        mwnn_graph_->mwnn_graph_nodes[mwnn_node.get_name()] = std::move(op_node);
        global_output_name = node_outputs[0].c_str();
      }
    }
    // Graph input and output handling
    auto &nodes = F->getNodes();
    auto &first_node = nodes.front();
    auto &last_node = nodes.back();
    auto input_name = std::string(first_node.getNthInput(0).getNode()->getName());
    auto output_name = std::string(last_node.getNthResult(0).getNode()->getName());
    for (auto &V : F->getParent()->getPlaceholders()) {
      if (!usedInFunction(V, F)) {
        continue;
      }
      auto glow_dims = V->getType()->dims();
      auto data_type = V->getType()->getElementType();
      int size = glow_dims.size();
      std::vector<int> dims(size);
      // Input dims from NCHW to NHWC
      dims[1] = int(glow_dims[3]);
      dims[3] = int(glow_dims[1]);
      dims[2] = int(glow_dims[2]);
      dims[0] = int(glow_dims[0]);
      if (getOutputSave(F, V)) {
        mwnn_graph_->set_graph_op_names(global_output_name);
        //Fills Graph Output Tensor Details - Name, Dims
        MWNNTensor mwnn_op_tensor(global_output_name, get_mwnn_type_glow(data_type), dims);
        mwnn_graph_->set_graph_op_tensor(mwnn_op_tensor);
      }
      else if(V->getName().equals(input_name)) {
        mwnn_graph_->set_graph_ip_names(V->getName());
        //Fills Graph Input Tensor Details - Name, Dims
        MWNNTensor mwnn_ip_tensor(V->getName(), get_mwnn_type_glow(data_type), dims);
        mwnn_graph_->set_graph_ip_tensor(mwnn_ip_tensor);
      }
    }
    optimizer::PassManager manager;
    if(CHW_TO_HWC)
    {
      for (auto g_t : mwnn_graph_->get_graph_ip_tensor()) {
        if(g_t.get_dims().size() == 4) {
          /*std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";*/
          optimizer::ConvertLayout cl(mwnn_graph_, g_t, CHW_TO_HWC, 0, false);
          manager.register_pass(cl);
        }
      }
    }
    if(HWC_TO_CHW)
    {
      for (auto g_t : mwnn_graph_->get_graph_initializers()) {
        if(g_t.get_dims().size() == 4) {
          /*std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";*/
          ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph_, g_t, 0, HWC_TO_CHW, true);
          manager.register_pass(cl);
        }
      }
      //Subgraph from other backends is already in CHW order
      if(graph_count == 1) {
        for (auto g_t : mwnn_graph_->get_graph_ip_tensor()) {
          if(g_t.get_dims().size() == 4) {
            /*std::cout << "\n Name : " << g_t.get_name();
            std::cout << "\t Dims : ";
            for (auto dim : g_t.get_dims())
              std::cout << dim << ",";*/
            ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph_, g_t, 0, HWC_TO_CHW, false);
            manager.register_pass(cl);
          }
        }
      }
    }
    auto mwnn_nodes = mwnn_graph_->get_graph_nodes();
    for (int node_idx = 0; node_idx < mwnn_graph_->get_graph_nodes().size(); node_idx++) {
      auto g_n = mwnn_nodes[node_idx];
      if(g_n.get_op_type() == "Relu") {
        optimizer::FuseRelu fr(mwnn_graph_, g_n);
        //std::cout << "\n MetaWareNNCC : " << fr.get_name();
        manager.register_pass(fr);
      }
      else if(g_n.get_op_type() == "Transpose") {
        optimizer::RemoveTranspose rt(mwnn_graph_, g_n);
        //std::cout << "\n MetaWareNNCC : " << rt.get_name();
        manager.register_pass(rt);
      }
    }
    optimizer::CalculateOffset co(mwnn_graph_);
    manager.register_pass(co);
    manager.run_passes();
    mwnn_exe_graph_ = std::make_shared<metawarenn::MWNNExecutableGraph>(*mwnn_graph_);

    #if INVOKE_NNAC
      std::cout << "\n ---------------------------Graph----------------------------- \n";
      std::cout << "\n Graph Name : " << mwnn_graph_->get_name();
    ::MWNN::MWNNGraphProto mwnn_graph_proto;
    mwnn_graph_proto.set_name(mwnn_graph_->get_name());
    for (auto g_ip : mwnn_graph_->get_graph_ip_names())
      mwnn_graph_proto.add_ip_name((g_ip));
    for (auto g_op : mwnn_graph_->get_graph_op_names())
      mwnn_graph_proto.add_op_name((g_op));

    std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
    for (auto g_ip : mwnn_graph_->get_graph_ip_tensor()) {
      std::cout << "\n Input Name : " << g_ip.get_name();
      std::cout << "\n Data Type : " << g_ip.get_type();
      std::cout << "\n Input Dims : ";
      auto input = mwnn_graph_proto.add_input();
      input->set_name(g_ip.get_name());
      input->set_type(g_ip.get_type());
      for (auto dim : g_ip.get_dims()) {
        std::cout << dim << ",";
        input->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
    for (auto g_op : mwnn_graph_->get_graph_op_tensor()) {
      std::cout << "\n Output Name : " << g_op.get_name();
      std::cout << "\n Data Type : " << g_op.get_type();
      std::cout << "\n Output Dims : ";
      auto output = mwnn_graph_proto.add_output();
      output->set_name(g_op.get_name());
      output->set_type(g_op.get_type());
      for (auto dim : g_op.get_dims()) {
        std::cout << dim << ",";
        output->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Nodes-------------------------- \n";
    for (auto g_n : mwnn_graph_->get_graph_nodes()) {
      std::cout << "\n ================================================================ \n";
      std::cout << "\n Node Name : " << g_n.get_name();
      std::cout << "\n Op Type : " << g_n.get_op_type();
      auto node = mwnn_graph_proto.add_node();
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
    for (auto g_t : mwnn_graph_->get_graph_initializers()) {
      auto initializer = mwnn_graph_proto.add_initializer();
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

    std::string g_name = mwnn_graph_->get_name();
    char* mwnn_op_path = nullptr;
    mwnn_op_path = getenv("NNAC_DUMPS_PATH");
    if(!IsPathExist(std::string(mwnn_op_path))) {
      int check = mkdir(mwnn_op_path, 0777);
      if(check != 0) {
        std::cout << "\nPlease check the directory path to store the serialized binary!!!!!";
        exit(1);
      }
    }
    auto mwnn_proto_bin = std::string(mwnn_op_path) + std::string(g_name) + ".bin";

    int fp = open(mwnn_proto_bin.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::cout << fp;
    std::cout << mwnn_graph_proto.SerializeToFileDescriptor(fp);
    close(fp);

    char* mwnn_lib_path = nullptr;
    mwnn_lib_path = getenv("METAWARENN_LIB_PATH");
    if(!IsPathExist(std::string(mwnn_lib_path)))
      std::cout << "\nPlease check the MetaWareNN Library path!!!";
    std::cout << "\n\n=================Initiating NNAC python script via shell script======================\n";
    std::string cmd = "bash " + std::string(mwnn_lib_path) +"/mwnnconvert/mwnn_convert.sh " + mwnn_proto_bin + " " + mwnn_op_path + " " + g_name + " " + std::to_string(graph_count);
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
    graph_outputs[mwnn_graph_->get_graph_op_names()[0]] = (float*)tensor->getUnsafePtr();

  }

  // **************************************** Calls to invoke the MetaWareNN Inference API ************************************
  MWNNInferenceApi mwapi;

  std::vector<std::string> ip_names = mwnn_graph_->get_graph_ip_names();
  auto ip_shape = mwnn_graph_->get_graph_ip_tensor()[0].get_dims();

  mwapi.prepareInput(graph_inputs[ip_names[0]], ip_shape);

  std::vector<std::string> op_names = mwnn_graph_->get_graph_op_names();
  auto op_shape = mwnn_graph_->get_graph_op_tensor()[0].get_dims();

  mwapi.prepareOutput(op_shape);

  mwapi.prepareGraph(mwnn_graph_->get_name());

  mwapi.runGraph();

  mwapi.getOutput(graph_outputs[op_names[0]], op_shape);

  // ******************************************* Call to invoke the local run function *****************************************

  //convert_to_mwnn_format(*mwnn_graph_, graph_inputs, graph_outputs, CHW_TO_HWC);
  return Error::success();
}
} // namespace metawarenn
