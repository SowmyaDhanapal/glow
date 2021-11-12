file(GLOB_RECURSE METAWARENN_LIB_SRC metawarenn_lib/*.cc)

add_library(MetaWareNN
            MetaWareNN.cpp
            MetaWareNNFactory.cpp
            MetaWareNNFunction.cpp
            MetaWareNNDeviceManager.cpp
            ${METAWARENN_LIB_SRC})

set(PROTOBUF_LIB "${CMAKE_CURRENT_SOURCE_DIR}/metawarenn_lib/lib/libprotobuf.so")
target_link_libraries(MetaWareNN
                      Backend
                      -lrt
                      ${PROTOBUF_LIB})

set(linked_backends ${linked_backends} MetaWareNN PARENT_SCOPE)
