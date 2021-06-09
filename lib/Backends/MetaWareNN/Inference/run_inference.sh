unzip mwnn_glow_inference_lib.zip
g++ -c ImageClassifier.cpp -o classifier
g++ -std=c++11 ./classifier ./libraries/obj/Loader.cpp.o ./libraries/obj/ExecutorCore.cpp.o ./libraries/obj/ExecutorCoreHelperFunctions.cpp.o ./libraries/obj/TaggedList.cpp.o -o inference ./libraries/obj/MetaWareNNFactory.cpp.o -L./libraries/ -lglow -L/usr/local/lib -lprotobuf -L/usr/local/lib -lprotobuf-lite -L/usr/lib/x86_64-linux-gnu -lglog -pthread -lgflags -ltinfo -lrt -lpng -ldl -ldouble-conversion -lLLVM-8 -L../../../../../glow/build_Release/thirdparty/folly -lfolly -L./libraries/ -lQuantization  -lQuantizationBase -lIROptimizerPipeline -lGraphOptimizer -lLower
