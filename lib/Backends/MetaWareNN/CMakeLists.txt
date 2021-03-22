add_library(MetaWareNN
            MetaWareNN.cpp
            MetaWareNNFactory.cpp
            DummyFunction.cpp)

target_link_libraries(MetaWareNN
                      Backend)                  

set(linked_backends ${linked_backends} MetaWareNN PARENT_SCOPE)