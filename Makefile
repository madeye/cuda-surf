###############################################################################
#
# CUDA-SURF v0.5
# Copyright 2010 FUDAN UNIVERSITY
# Author: Max Lv
# Email: max.c.lv#gmail.com
#
################################################################################

# Add source files here
EXECUTABLE	:= cudasurf
# CUDA source files (compiled with cudacc)
CUFILES		:= \
	main.cu

CCFILES     := \
	lfsr.cpp

#SMVERSIONFLAGS += -arch sm_11

COMMONFLAGS  += `pkg-config --cflags opencv`

#NVCCFLAGS += -ptx
LINKFLAGS += `pkg-config --libs opencv` -lzmq

################################################################################
include common.mk
