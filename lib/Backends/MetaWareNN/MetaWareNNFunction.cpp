#include "MetaWareNNFunction.h"

namespace metawarenn {


MetaWareNNFunction::MetaWareNNFunction(runtime::RuntimeBundle &&bundle, Function *F)
    : CompiledFunction(std::move(bundle)) {
    findIOPlaceholders(F);
    graph_count++;
    std::string subgraph_name = "MetaWareNN_" + std::to_string(graph_count);

    /*Create MetaWareNN High Level Graph Representation from Glow SubGraph Function*/
    graph_ = std::make_shared<Graph>();
    graph_->set_name(subgraph_name);

    std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
    std::cout << "\n MWNN Graph Name : " << graph_->get_name();

    GraphPostOrderVisitor visitor(*F);
    auto node_list = visitor.getPostOrder();
    auto global_output_name = "";

    for (auto *node : node_list) {
        LOG(INFO) << "==============================================================================================================";
        std::string node_name;
        std::string node_op_type;
        std::vector<std::string> node_inputs;
        std::vector<std::string> node_outputs;
        std::vector<::metawarenn::Attribute> node_attributes;
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
            graph_->initializer_names.insert(filter_name);
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
            metawarenn::Tensor weight_tensor(filter_name, weight_dims, get_mwnn_type_glow(data_type), weights);
            graph_->set_graph_initializers(weight_tensor);
            auto bias_node_value = conv_node->getBias();
            auto bias_name = bias_node_value.generateNodeOutputName(true);
            // Check to avoid redundant constants in initializers
            if(!graph_->initializer_names.count(bias_name))
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
              graph_->initializer_names.insert(bias_name);
              type = bias_tensor.getType();
              data_type = type.getElementType();
              metawarenn::Tensor m_bias_tensor(bias_name, bias_dims, get_mwnn_type_glow(data_type), bias);
              graph_->set_graph_initializers(m_bias_tensor);
            }
            auto dilations = conv_node->getDilation();
            auto strides = conv_node->getStrides();
            auto pads = conv_node->getPads();
            auto group = conv_node->getGroup();
            metawarenn::Attribute attr_dilate("dilations", std::vector<int>{int(dilations[0]), int(dilations[1])});
            node_attributes.emplace_back(attr_dilate);
            metawarenn::Attribute attr_group("group", std::vector<int>{int(group)});
            node_attributes.emplace_back(attr_group);
            metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int>{(int)filterDims.h, (int)filterDims.w});
            node_attributes.emplace_back(attr_kernel_shape);
            metawarenn::Attribute attr_pad("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
            node_attributes.emplace_back(attr_pad);
            metawarenn::Attribute attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
            node_attributes.emplace_back(attr_stride);
            metawarenn::Attribute attr_act("activation", std::vector<int>{0});
            node_attributes.emplace_back(attr_act);
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
          node_op_type = "AveragePool";
          auto *avgpool_node = llvm::cast<AvgPoolNode>(node);
          auto kernels = avgpool_node->getKernels();
          auto strides = avgpool_node->getStrides();
          auto pads = avgpool_node->getPads();
          auto count_include_pad = avgpool_node->getCountIncludePads();
          metawarenn::Attribute attr_count_include_pad("count_include_pad", std::vector<int>{count_include_pad});
          node_attributes.emplace_back(attr_count_include_pad);
          metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int>{int(kernels[0]), int(kernels[1])});
          node_attributes.emplace_back(attr_kernel_shape);
          metawarenn::Attribute attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
          node_attributes.emplace_back(attr_stride);
          metawarenn::Attribute attr_pads("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
          node_attributes.emplace_back(attr_pads);
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
          auto perm = transpose_node->getShuffle();
          metawarenn::Attribute attr_pads("perm", std::vector<int>{int(perm[0]), int(perm[1]), int(perm[2]), int(perm[3])});
          node_attributes.emplace_back(attr_pads);
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
          metawarenn::Tensor reshape_tensor(initializer_name, dims_, get_mwnn_type_glow(ElemKind::Int64ITy), dims_vec);
          graph_->set_graph_initializers(reshape_tensor);
          node_inputs.emplace_back(input_name);
          node_inputs.emplace_back(initializer_name);
          graph_->initializer_names.insert(initializer_name);
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
          metawarenn::Attribute attr_alpha("alpha", std::vector<int>{int(lrn_node->getAlpha())});
          node_attributes.emplace_back(attr_alpha);
          metawarenn::Attribute attr_beta("beta", std::vector<int>{int(lrn_node->getBeta())});
          node_attributes.emplace_back(attr_beta);
          metawarenn::Attribute attr_size("size", std::vector<int>{int(lrn_node->getHalfWindowSize())});
          node_attributes.emplace_back(attr_size);
          metawarenn::Attribute attr_bias("bias", std::vector<int>{int(lrn_node->getK())});
          node_attributes.emplace_back(attr_bias);
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
          metawarenn::Attribute attr_dilations("dilations", std::vector<int>{1,1});
          node_attributes.emplace_back(attr_dilations);
          metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int>{int(kernels[0]), int(kernels[1])});
          node_attributes.emplace_back(attr_kernel_shape);
          metawarenn::Attribute attr_pad("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
          node_attributes.emplace_back(attr_stride);
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
          graph_->initializer_names.insert(filter_name);
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
          metawarenn::Tensor weight_tensor(filter_name, weight_dims, get_mwnn_type_glow(data_type), weights);
          graph_->set_graph_initializers(weight_tensor);
          auto bias_node_value = gemm_node->getNthInput(2);
          auto bias_name = bias_node_value.generateNodeOutputName(true);
          LOG(INFO) << "bias_name: " << bias_name;
          node_inputs.emplace_back(bias_name);
          graph_->initializer_names.insert(bias_name);
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
          metawarenn::Tensor m_bias_tensor(bias_name, bias_dims, get_mwnn_type_glow(data_type), bias);
          graph_->set_graph_initializers(m_bias_tensor);
          metawarenn::Attribute attr_alpha("alpha", std::vector<int>{int(gemm_node->getAlpha())});
          node_attributes.emplace_back(attr_alpha);
          metawarenn::Attribute attr_beta("beta", std::vector<int>{int(gemm_node->getBeta())});
          node_attributes.emplace_back(attr_beta);
          metawarenn::Attribute attr_transA("transA", std::vector<int>{int(gemm_node->getTransposeA())});
          node_attributes.emplace_back(attr_transA);
          metawarenn::Attribute attr_transB("transB", std::vector<int>{int(gemm_node->getBeta())});
          node_attributes.emplace_back(attr_transB);
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
          metawarenn::Attribute attr_axis("axis", std::vector<int>{int(concat_node->getDim())});
          node_attributes.emplace_back(attr_axis);
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
          graph_->initializer_names.insert(bias_name);
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
          metawarenn::Tensor m_bias_tensor(bias_name, bias_dims, get_mwnn_type_glow(data_type), bias);
          graph_->set_graph_initializers(m_bias_tensor);
          metawarenn::Attribute attr_epsilon("epsilon", std::vector<float>{batchnorm_node->getEpsilon()});
          node_attributes.emplace_back(attr_epsilon);
          metawarenn::Attribute attr_momentum("momentum", std::vector<float>{batchnorm_node->getMomentum()});
          node_attributes.emplace_back(attr_momentum);
          auto input_name = batchnorm_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          auto output_name = batchnorm_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ChannelShuffleNodeKind: //Check for onnx conversion
        {
          node_op_type = "ChannelShuffle";
          auto *channel_shuffle_node = llvm::cast<ChannelShuffleNode>(node);
          metawarenn::Attribute attr_group("group", std::vector<int>{int(channel_shuffle_node->getGroup())});
          node_attributes.emplace_back(attr_group);
          metawarenn::Attribute attr_kernel("kernel", std::vector<int>{int(channel_shuffle_node->getKernel())});
          node_attributes.emplace_back(attr_kernel);
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
          metawarenn::Tensor min_tensor("min", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{(float)clip_node->getMin()});
          graph_->set_graph_initializers(min_tensor);
          node_inputs.emplace_back(min_tensor.get_name());
          metawarenn::Tensor max_tensor("max", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{(float)clip_node->getMax()});
          graph_->set_graph_initializers(max_tensor);
          node_inputs.emplace_back(min_tensor.get_name());
          auto input_name = clip_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          auto output_name = clip_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::FullyConnectedNodeKind:
        {
          node_op_type = "Gemm";
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
          graph_->initializer_names.insert(filter_name);
          node_inputs.emplace_back(filter_name);
          metawarenn::Tensor weight_tensor(filter_name, weight_dims, get_mwnn_type_glow(data_type), weights);
          graph_->set_graph_initializers(weight_tensor);
          auto bias_node_value = fc_node->getBias();;
          auto bias_name = bias_node_value.generateNodeOutputName(true);
          LOG(INFO) << "bias_name: " << bias_name;
          node_inputs.emplace_back(bias_name);
          graph_->initializer_names.insert(bias_name);
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
          metawarenn::Tensor m_bias_tensor(bias_name, bias_dims, get_mwnn_type_glow(data_type), bias);
          graph_->set_graph_initializers(m_bias_tensor);
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
        case Kinded::Kind::SliceNodeKind:
        {
          node_op_type = "Slice";
          auto *slice_node = llvm::cast<SliceNode>(node);
          auto starts = slice_node->getStart();
          auto outs = slice_node->getResult().dims();
          auto size = starts.size();
          std::vector<float> starts_vec(size);
          std::vector<float> ends_vec(size);
          std::vector<float> axes_vec(size);
          std::vector<float> steps_vec(size);

          for (unsigned b = 0, e = size; b < e; ++b) {
            starts_vec[b] = (float)starts[b];
            starts_vec[b] = (float)outs[b] + starts[b];
            axes_vec[b] = (float)b; //  If axes are omitted, they are set to [0, ..., ndim-1]
            steps_vec[b] = (float)0;
          }
          metawarenn::Tensor tensor_starts(node_name + "_starts", std::vector<int>{int(size)}, ElementType::element_type::float_, starts_vec);
          graph_->set_graph_initializers(tensor_starts);
          node_inputs.emplace_back(tensor_starts.get_name());
          metawarenn::Tensor tensor_ends(node_name + "ends", std::vector<int>{int(size)}, ElementType::element_type::float_, ends_vec);
          graph_->set_graph_initializers(tensor_ends);
          node_inputs.emplace_back(tensor_ends.get_name());
          metawarenn::Tensor tensor_axes(node_name + "axes", std::vector<int>{int(size)}, ElementType::element_type::float_, axes_vec);
          graph_->set_graph_initializers(tensor_axes);
          node_inputs.emplace_back(tensor_axes.get_name());
          metawarenn::Tensor tensor_steps(node_name + "steps", std::vector<int>{int(size)}, ElementType::element_type::float_, steps_vec);
          graph_->set_graph_initializers(tensor_steps);
          node_inputs.emplace_back(tensor_steps.get_name());
          auto input_name = slice_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = slice_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::PowNodeKind:
        {
          node_op_type = "Pow";
          auto *pow_node = llvm::cast<PowNode>(node);
          auto input_name = pow_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = pow_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::TopKNodeKind:
        {
          node_op_type = "TopK";
          auto *topk_node = llvm::cast<TopKNode>(node);
          auto k_val = topk_node->getK();
          metawarenn::Tensor k_tensor("K", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{float(k_val)});
          graph_->set_graph_initializers(k_tensor);
          node_inputs.emplace_back(k_tensor.get_name());
          auto input_name = topk_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = std::string(topk_node->getOutputName(0));
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ArgMaxNodeKind:
        {
          node_op_type = "ArgMax";
          auto *argmax_node = llvm::cast<ArgMaxNode>(node);
          metawarenn::Attribute attr_axis("axis", std::vector<int>{int(argmax_node->getAxis())});
          metawarenn::Attribute attr_keep_dims("keepDims", std::vector<int>{int(argmax_node->getKeepDims())});
          node_attributes.emplace_back(attr_axis);
          auto input_name = argmax_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = argmax_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ArgMinNodeKind:
        {
          node_op_type = "ArgMin";
          auto *argmin_node = llvm::cast<ArgMinNode>(node);
          metawarenn::Attribute attr_axis("axis", std::vector<int>{int(argmin_node->getAxis())});
          node_attributes.emplace_back(attr_axis);
          metawarenn::Attribute attr_keep_dims("keepDims", std::vector<int>{int(argmin_node->getKeepDims())});
          node_attributes.emplace_back(attr_keep_dims);
          auto input_name = argmin_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = argmin_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::PReluNodeKind:
        {
          node_op_type = "PRelu";
          auto *prelu_node = llvm::cast<PReluNode>(node);
          auto input_name = prelu_node->getInputName(0);
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
          }
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = prelu_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::GatherNodeKind:
        {
          node_op_type = "Gather";
          auto *gather_node = llvm::cast<GatherNode>(node);
          auto batch_dims = gather_node->getBatchDims();
          metawarenn::Attribute attr_axis("axis", std::vector<int>{(int)batch_dims});
          node_attributes.emplace_back(attr_axis);
          NodeValue indices = gather_node->getIndices();
          std::vector<int> ind_dims;
          int i = 0;
          for(auto dim: indices.dims())
            ind_dims[i] = dim;
          if (Constant *c = llvm::dyn_cast<Constant>(indices.getNode())) {
            auto handle = c->getHandle<int>();
            auto begin = &handle.raw(0);
            std::vector<float> data(begin, begin + handle.actualSize());
            metawarenn::Tensor indices_tensor(indices.getNode()->getName(), ind_dims, ElementType::element_type::float_, data);
            graph_->set_graph_initializers(indices_tensor);
          node_inputs.emplace_back(indices_tensor.get_name());
          }
          auto input_name = gather_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = gather_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::MulNodeKind:
        {
          node_op_type = "Mul";
          auto *mul_node = llvm::cast<MulNode>(node);
          auto input1 = mul_node->getLHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 1: " << input1;
          auto input2 = mul_node->getRHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 2: " << input2;
          node_inputs.emplace_back(input1);
          node_inputs.emplace_back(input2);
          auto output_name = mul_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::DivNodeKind:
        {
          node_op_type = "Div";
          auto *div_node = llvm::cast<DivNode>(node);
          auto input1 = div_node->getLHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 1: " << input1;
          auto input2 = div_node->getRHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 2: " << input2;
          node_inputs.emplace_back(input1);
          node_inputs.emplace_back(input2);
          auto output_name = div_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::SubNodeKind:
        {
          node_op_type = "Sub";
          auto *sub_node = llvm::cast<SubNode>(node);
          auto input1 = sub_node->getLHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 1: " << input1;
          auto input2 = sub_node->getRHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 2: " << input2;
          node_inputs.emplace_back(input1);
          node_inputs.emplace_back(input2);
          auto output_name = sub_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::AbsNodeKind:
        {
          node_op_type = "Abs";
          auto *abs_node = llvm::cast<AbsNode>(node);
          auto input_name = abs_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = abs_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::AndNodeKind:
        {
          node_op_type = "And";
          auto *and_node = llvm::cast<AndNode>(node);
          auto input_name = and_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = and_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ExpNodeKind:
        {
          node_op_type = "Exp";
          auto *exp_node = llvm::cast<ExpNode>(node);
          auto input_name = exp_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = exp_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::MaxNodeKind:
        {
          node_op_type = "Max";
          auto *max_node = llvm::cast<MaxNode>(node);
          auto input_name = max_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = max_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::MinNodeKind:
        {
          node_op_type = "Min";
          auto *min_node = llvm::cast<MinNode>(node);
          auto input_name = min_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = min_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::PadNodeKind:
        {
          node_op_type = "Pad";
          auto *pad_node = llvm::cast<PadNode>(node);
          auto pads = pad_node->getPads();
          metawarenn::Tensor pads_tensor(node_name + "_pads", std::vector<int>{(int)pads.size()}, ElementType::element_type::float_, std::vector<float>{(float)pads[0], (float)pads[1], (float)pads[2], (float)pads[3]});
          node_inputs.emplace_back(pads_tensor.get_name());
          metawarenn::Tensor value_tensor(node_name + "_value", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{(float)pad_node->getValue()});
          node_inputs.emplace_back(value_tensor.get_name());
          metawarenn::Attribute attr_mode("mode", std::vector<int>{int(pad_node->getMode())});
          node_attributes.emplace_back(attr_mode);
          auto input_name = pad_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = pad_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::CeilNodeKind:
        {
          node_op_type = "Ceil";
          auto *ceil_node = llvm::cast<CeilNode>(node);
          auto input_name = ceil_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = ceil_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::FloorNodeKind:
        {
          node_op_type = "Floor";
          auto *floor_node = llvm::cast<FloorNode>(node);
          auto input_name = floor_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = floor_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::SwishNodeKind:
        {
          node_op_type = "HardSwish";
          auto *swish_node = llvm::cast<SwishNode>(node);
          auto input_name = swish_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = swish_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::LeakyReluNodeKind:
        {
          node_op_type = "LeakyRelu";
          auto *lrelu_node = llvm::cast<LeakyReluNode>(node);
          metawarenn::Attribute attr_alpha("alpha", std::vector<int>{(int)lrelu_node->getAlpha()});
          node_attributes.emplace_back(attr_alpha);
          auto input_name = lrelu_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = lrelu_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::LogNodeKind:
        {
          node_op_type = "Log";
          auto *log_node = llvm::cast<LogNode>(node);
          auto input_name = log_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = log_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::MatMulNodeKind:
        {
          node_op_type = "MatMul";
          auto *matmul_node = llvm::cast<MatMulNode>(node);
          auto input1 = matmul_node->getLHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 1: " << input1;
          auto input2 = matmul_node->getRHS().generateNodeOutputName(true);
          LOG(INFO) << "input_name 2: " << input2;
          node_inputs.emplace_back(input1);
          node_inputs.emplace_back(input2);
          auto output_name = matmul_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::NonMaxSuppressionNodeKind:
        {
          node_op_type = "NonMaxSuppression";
          auto *nms_node = llvm::cast<NonMaxSuppressionNode>(node);
          metawarenn::Attribute attr_cpb("center_point_box", std::vector<int>{(int)nms_node->getCenterPointBox()});
          node_attributes.emplace_back(attr_cpb);
          NodeValue boxes = nms_node->getBoxes();
          NodeValue scores = nms_node->getScores();
          std::vector<int> box_dims(boxes.dims().size());
          std::vector<int> scores_dims(scores.dims().size());
          int i = 0;
          for(auto dim: boxes.dims())
            box_dims[i] = dim;
          i = 0;
          for(auto dim: scores.dims())
            scores_dims[i] = dim;
          if (Constant *c = llvm::dyn_cast<Constant>(boxes.getNode())) {
            auto handle = c->getHandle<float>();
            auto begin = &handle.raw(0);
            std::vector<float> data(begin, begin + handle.actualSize());
            metawarenn::Tensor boxes_tensor(boxes.getNode()->getName(), box_dims, ElementType::element_type::float_, data);
            graph_->set_graph_initializers(boxes_tensor);
            node_inputs.emplace_back(boxes_tensor.get_name());
          }
          if (Constant *c = llvm::dyn_cast<Constant>(scores.getNode())) {
            auto handle = c->getHandle<float>();
            auto begin = &handle.raw(0);
            std::vector<float> data(begin, begin + handle.actualSize());
            metawarenn::Tensor scores_tensor(scores.getNode()->getName(), scores_dims, ElementType::element_type::float_, data);
            graph_->set_graph_initializers(scores_tensor);
            node_inputs.emplace_back(scores_tensor.get_name());
          }
          metawarenn::Tensor max_out_box_tensor(node_name + "_max_out_box", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{float(nms_node->getMaxOutputBoxesPerClass())});
          graph_->set_graph_initializers(max_out_box_tensor);
          node_inputs.emplace_back(max_out_box_tensor.get_name());
          metawarenn::Tensor iou_thresh_tensor(node_name + "_iou_thresh", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{nms_node->getIouThreshold()});
          graph_->set_graph_initializers(iou_thresh_tensor);
          node_inputs.emplace_back(iou_thresh_tensor.get_name());
          metawarenn::Tensor score_threshold_tensor(node_name + "_score_thresh", std::vector<int>{1}, ElementType::element_type::float_, std::vector<float>{nms_node->getScoreThreshold()});
          graph_->set_graph_initializers(score_threshold_tensor);
          node_inputs.emplace_back(score_threshold_tensor.get_name());
          auto input_name = nms_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = nms_node->getOutputName(0).str();
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::NotNodeKind:
        {
          node_op_type = "Not";
          auto *not_node = llvm::cast<NotNode>(node);
          auto input_name = not_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = not_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::BatchedReduceMeanNodeKind:
        {
          node_op_type = "ReduceMean";
          auto *reduce_mean_node = llvm::cast<BatchedReduceMeanNode>(node);
          auto axes = reduce_mean_node->getAxes();
          std::vector<int> axes_vec(axes.size());
          int i = 0;
          for(auto ax: axes_vec)
            axes_vec[i++] = axes[i];
          metawarenn::Attribute attr_axes("axes", axes_vec);
          node_attributes.emplace_back(attr_axes);
          auto input_name = reduce_mean_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = reduce_mean_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::BatchedReduceMinNodeKind:
        {
          node_op_type = "ReduceMin";
          auto *reduce_min_node = llvm::cast<BatchedReduceMinNode>(node);
          auto axes = reduce_min_node->getAxes();
          std::vector<int> axes_vec(axes.size());
          int i = 0;
          for(auto ax: axes_vec)
            axes_vec[i++] = axes[i];
          metawarenn::Attribute attr_axes("axes", axes_vec);
          node_attributes.emplace_back(attr_axes);
          auto input_name = reduce_min_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = reduce_min_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::BatchedReduceMaxNodeKind:
        {
          node_op_type = "ReduceMax";
          auto *reduce_max_node = llvm::cast<BatchedReduceMaxNode>(node);
          auto axes = reduce_max_node->getAxes();
          std::vector<int> axes_vec(axes.size());
          int i = 0;
          for(auto ax: axes_vec)
            axes_vec[i++] = axes[i];
          metawarenn::Attribute attr_axes("axes", axes_vec);
          node_attributes.emplace_back(attr_axes);
          auto input_name = reduce_max_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = reduce_max_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::BatchedReduceSumSquareNodeKind:
        case Kinded::Kind::BatchedReduceAddNodeKind:
        {
          node_op_type = "ReduceSum";
          auto *reduce_add_node = llvm::cast<BatchedReduceAddNode>(node);
          metawarenn::Attribute attr_axes("axes", std::vector<int>{(int)reduce_add_node->getAxis()});
          node_attributes.emplace_back(attr_axes);
          auto input_name = reduce_add_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = reduce_add_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
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
          node_inputs.emplace_back(scales_tensor.get_name());
          metawarenn::Attribute attr_trans_mode("coordinate_transformation_mode", std::vector<std::string>{"asymmetric"});
          node_attributes.emplace_back(attr_trans_mode);
          metawarenn::Attribute attr_mode("mode", std::vector<std::string>{"nearest"});
          node_attributes.emplace_back(attr_mode);
          metawarenn::Attribute attr_near_mode("nearest_mode", std::vector<std::string>{"floor"});
          node_attributes.emplace_back(attr_near_mode);
          auto input_name = resize_near_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = resize_near_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
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
          node_inputs.emplace_back(scales_tensor.get_name());
          metawarenn::Attribute attr_trans_mode("coordinate_transformation_mode", std::vector<std::string>{"asymmetric"});
          node_attributes.emplace_back(attr_trans_mode);
          metawarenn::Attribute attr_mode("mode", std::vector<std::string>{"linear"});
          node_attributes.emplace_back(attr_mode);
          auto input_name = resize_bilinear_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = resize_bilinear_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::ROIAlignNodeKind:
        {
          node_op_type = "RoiAlign";
          auto *roi_align_node = llvm::cast<ROIAlignNode>(node);
          std::vector<int> roi_dims;
          std::vector<int> batch_ind_dims;
          NodeValue boxes = roi_align_node->getBoxes();
          NodeValue batch_indices = roi_align_node->getBatchIndices();
          int i = 0;
          for(auto dim: boxes.dims())
            roi_dims[i] = dim;
          i = 0;
          for(auto dim: batch_indices.dims())
            batch_ind_dims[i] = dim;
          if (Constant *c = llvm::dyn_cast<Constant>(boxes.getNode())) {
            auto handle = c->getHandle<int>();
            auto begin = &handle.raw(0);
            std::vector<float> data(begin, begin + handle.actualSize());
            metawarenn::Tensor rois_tensor(boxes.getNode()->getName(), roi_dims, ElementType::element_type::float_, data);
            graph_->set_graph_initializers(rois_tensor);
            node_inputs.emplace_back(rois_tensor.get_name());
          }
          if (Constant *c = llvm::dyn_cast<Constant>(batch_indices.getNode())) {
            auto handle = c->getHandle<int>();
            auto begin = &handle.raw(0);
            std::vector<float> data(begin, begin + handle.actualSize());
            metawarenn::Tensor batch_ind_tensor(batch_indices.getNode()->getName(), roi_dims, ElementType::element_type::float_, data);
            graph_->set_graph_initializers(batch_ind_tensor);
            node_inputs.emplace_back(batch_ind_tensor.get_name());
          }
          switch(roi_align_node->getMode()) {
            case PoolingMode::AVG: {
              metawarenn::Attribute attr_mode("mode", std::vector<std::string>{"avg"});
              node_attributes.emplace_back(attr_mode);
              break;
            }
            case PoolingMode::MAX: {
              metawarenn::Attribute attr_mode("mode", std::vector<std::string>{"max"});
              node_attributes.emplace_back(attr_mode);
              break;
            }
          }
          metawarenn::Attribute attr_out_h("output_height", std::vector<int>{(int)roi_align_node->getOutputHeight()});
          node_attributes.emplace_back(attr_out_h);
          metawarenn::Attribute attr_out_w("output_width", std::vector<int>{(int)roi_align_node->getOutputWidth()});
          node_attributes.emplace_back(attr_out_w);
          metawarenn::Attribute attr_s_ratio("sampling_ratio", std::vector<int>{(int)roi_align_node->getSamplingRatio()});
          node_attributes.emplace_back(attr_s_ratio);
          metawarenn::Attribute attr_s_scale("spatial_scale", std::vector<float>{(float)roi_align_node->getSpatialScale()});
          node_attributes.emplace_back(attr_s_scale);
          auto input_name = roi_align_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = roi_align_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::RoundNodeKind:
        {
          node_op_type = "Round";
          auto *round_node = llvm::cast<RoundNode>(node);
          auto input_name = round_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = round_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::SigmoidNodeKind:
        {
          node_op_type = "Sigmoid";
          auto *sigmoid_node = llvm::cast<SigmoidNode>(node);
          auto input_name = sigmoid_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = sigmoid_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::SpaceToDepthNodeKind:
        {
          node_op_type = "SpaceToDepth";
          auto *space_depth_node = llvm::cast<SpaceToDepthNode>(node);
          metawarenn::Attribute attr_b_size("blocksize", std::vector<int>{(int)space_depth_node->getBlockSize()});
          node_attributes.emplace_back(attr_b_size);
          auto input_name = space_depth_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = space_depth_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::SqrtNodeKind:
        {
          node_op_type = "Sqrt";
          auto *sqrt_node = llvm::cast<SqrtNode>(node);
          auto input_name = sqrt_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = sqrt_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::TanhNodeKind:
        {
          node_op_type = "Tanh";
          auto *tanh_node = llvm::cast<TanhNode>(node);
          auto input_name = tanh_node->getInputName(0);
          node_inputs.emplace_back(input_name);
          LOG(INFO) << "input_name: " << input_name;
          auto output_name = tanh_node->getResult().generateNodeOutputName(true);
          node_outputs.emplace_back(output_name);
          LOG(INFO) << "output_name: " << output_name;
          break;
        }
        default:
          break;
        }
        metawarenn::Node m_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
        graph_->set_graph_nodes(m_node);
        auto op_node = m_node.get_node();
        graph_->graph_nodes[m_node.get_name()] = std::move(op_node);
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
        graph_->set_graph_op_names(global_output_name);
        //Fills Graph Output Tensor Details - Name, Dims
        Tensor op_tensor(global_output_name, get_mwnn_type_glow(data_type), dims);
        graph_->set_graph_op_tensor(op_tensor);
      }
      else if(V->getName().equals(input_name)) {
        graph_->set_graph_ip_names(V->getName());
        //Fills Graph Input Tensor Details - Name, Dims
        Tensor ip_tensor(V->getName(), get_mwnn_type_glow(data_type), dims);
        graph_->set_graph_ip_tensor(ip_tensor);
      }
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
          optimizer::ConvertLayout cl(graph_, g_t, CHW_TO_HWC, 0, false);
          manager.register_pass(cl);
        }
      }
    }
    if(HWC_TO_CHW)
    {
      for (auto g_t : graph_->get_graph_initializers()) {
        if(g_t.get_dims().size() == 4) {
          /*std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";*/
          ::metawarenn::optimizer::ConvertLayout cl(graph_, g_t, 0, HWC_TO_CHW, true);
          manager.register_pass(cl);
        }
      }
      //Subgraph from other backends is already in CHW order
      if(graph_count == 1) {
        for (auto g_t : graph_->get_graph_ip_tensor()) {
          if(g_t.get_dims().size() == 4) {
            /*std::cout << "\n Name : " << g_t.get_name();
            std::cout << "\t Dims : ";
            for (auto dim : g_t.get_dims())
              std::cout << dim << ",";*/
            ::metawarenn::optimizer::ConvertLayout cl(graph_, g_t, 0, HWC_TO_CHW, false);
            manager.register_pass(cl);
          }
        }
      }
    }
    auto m_nodes = graph_->get_graph_nodes();
    for (int node_idx = 0; node_idx < graph_->get_graph_nodes().size(); node_idx++) {
      auto g_n = m_nodes[node_idx];
      if(g_n.get_op_type() == "Relu") {
        optimizer::FuseRelu fr(graph_, g_n);
        //std::cout << "\n MetaWareNNCC : " << fr.get_name();
        manager.register_pass(fr);
      }
      else if(g_n.get_op_type() == "Transpose") {
        optimizer::RemoveTranspose rt(graph_, g_n);
        //std::cout << "\n MetaWareNNCC : " << rt.get_name();
        manager.register_pass(rt);
      }
    }
    optimizer::CalculateOffset co(graph_);
    manager.register_pass(co);
    manager.run_passes();

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

    exe_graph_ = std::make_shared<metawarenn::ExecutableGraph>(*graph_);

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

  // ******************************************* Call to invoke the local run function *****************************************

  //convert_to_mwnn_format(*graph_, graph_inputs, graph_outputs, CHW_TO_HWC);
  return Error::success();
}
} // namespace metawarenn
