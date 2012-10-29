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
CUFILES		:= main.cu lfsr.cu
# CUDA dependency files
CU_DEPS		:= \
	integral_kernel.cu \
	det_kernel.cu \
	surf_kernel.cu \
	match_kernel.cu

#SMVERSIONFLAGS += -arch sm_11

CXXFLAGS  += -fopenmp
CFLAGS  += -fopenmp
COMMONFLAGS  += `pkg-config --cflags opencv`

NVCCFLAGS += -ptx
LINKFLAGS += `pkg-config --libs opencv`

################################################################################
include common.mk
