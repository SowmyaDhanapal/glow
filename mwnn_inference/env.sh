#!/bin/sh

########### Executable Networks flow ##############
#set the path to GLOW
export FRAMEWORK_PATH=/path/to/glow/
export METAWARENN_LIB_PATH=$FRAMEWORK_PATH"/lib/Backends/MetaWareNN/metawarenn_lib/"
export EXEC_DUMPS_PATH=$FRAMEWORK_PATH"/EXEC_DUMPS/"

########### NNAC - EV binary generation flow ##############
#set the path to ARC directory
export ARC_PATH=/path/to/ARC/
export EV_CNNMODELS_HOME=/path/to/cnn_tools/cnn_models/
export NNAC_DUMPS_PATH=$FRAMEWORK_PATH"/NNAC_DUMPS/"