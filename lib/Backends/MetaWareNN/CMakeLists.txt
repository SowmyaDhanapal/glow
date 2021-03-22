include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

add_library(MetaWareNN
            MetaWareNN.cpp
            MetaWareNNFactory.cpp
            DummyFunction.cpp
            MetaWareNNDeviceManager.cpp
            metawarenn_lib/metawarenn_model.cc
            metawarenn_lib/metawarenn_graph.cc
            metawarenn_lib/metawarenn_tensor.cc
            metawarenn_lib/metawarenn_node.cc
            metawarenn_lib/metawarenn_value_info.cc
            metawarenn_lib/metawarenn_attribute.cc
            metawarenn_lib/metawarenn_utils.cc
            metawarenn_lib/kernels/helpers/mli_helpers.cc
            metawarenn_lib/op/add.cc
            metawarenn_lib/op/avg_pool.cc
            metawarenn_lib/op/constant.cc
            metawarenn_lib/op/conv.cc
            metawarenn_lib/op/depthwise_conv.cc
            metawarenn_lib/op/input_data.cc
            metawarenn_lib/op/node.cc
            metawarenn_lib/op/relu.cc
            metawarenn_lib/op/reshape.cc
            metawarenn_lib/op/softmax.cc
            metawarenn_lib/optimizer/metawarenn_optimizer.cc
            metawarenn_lib/optimizer/remove_reshape.cc
            metawarenn_lib/optimizer/fuse_relu.cc
            metawarenn_lib/optimizer/convert_layout.cc
            metawarenn_lib/optimizer/pass_manager.cc
            metawarenn_lib/onnx/onnx-ml.pb.cc)

target_link_libraries(MetaWareNN
                      Backend
                      ${PROTOBUF_LIBRARY})

set(linked_backends ${linked_backends} MetaWareNN PARENT_SCOPE)