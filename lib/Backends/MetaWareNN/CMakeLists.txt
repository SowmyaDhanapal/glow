add_library(MetaWareNN
            MetaWareNN.cpp
            MetaWareNNFactory.cpp
            MetaWareNNFunction.cpp
            MetaWareNNDeviceManager.cpp
            metawarenn_lib/metawarenn_graph.cc
            metawarenn_lib/metawarenn_tensor.cc
            metawarenn_lib/metawarenn_node.cc
            metawarenn_lib/metawarenn_attribute.cc
            metawarenn_lib/optimizer/metawarenn_optimizer.cc
            metawarenn_lib/optimizer/remove_reshape.cc
            metawarenn_lib/optimizer/fuse_relu.cc
            metawarenn_lib/optimizer/convert_layout.cc
            metawarenn_lib/optimizer/remove_transpose.cc
            metawarenn_lib/optimizer/pass_manager.cc
            metawarenn_lib/optimizer/calculate_offset.cc
            metawarenn_lib/optimizer/fuse_batchnorm.cc
            metawarenn_lib/optimizer/expand_dimension.cc
            metawarenn_lib/mwnnconvert/mwnn_protobuf/cpp_wrapper/MWNN.pb.cc
            metawarenn_lib/mwnnconvert/onnx_protobuf/onnx-ml.pb.cc
            metawarenn_lib/mwnnconvert/mwnn_to_onnx_proto.cc
            metawarenn_lib/mwnnconvert/mwnn_to_proto.cc
            metawarenn_lib/executable_network/metawarenn_executable_graph.cc
            metawarenn_lib/executable_network/metawarenn_serialization.cc
            metawarenn_lib/inference_engine/mwnn_inference_engine.cc
            metawarenn_lib/inference_engine/mwnn_execution_context.cc
            metawarenn_lib/inference_engine/mwnn_device.cc
            metawarenn_lib/inference_engine/mwnn_builder.cc
            metawarenn_lib/inference_engine/mwnn_optimization_profile.cc
            metawarenn_lib/inference_engine/mwnn_encrypt_decrypt.cc)

set(PROTOBUF_LIB "${CMAKE_CURRENT_SOURCE_DIR}/metawarenn_lib/lib/libprotobuf.so")
target_link_libraries(MetaWareNN
                      Backend
                      -lrt
                      ${PROTOBUF_LIB})

set(linked_backends ${linked_backends} MetaWareNN PARENT_SCOPE)
