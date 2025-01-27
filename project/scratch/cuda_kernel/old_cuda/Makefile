CUDA_PATH ?= /usr/local/cuda
PYTHON_PATH ?= /home/bryan_diffuse_bio/miniconda/envs/core/include/python3.9
PYBIND11_PATH ?= /home/bryan_diffuse_bio/miniconda/envs/core/lib/python3.9/site-packages/pybind11/include
JAX_PATH ?= /home/bryan_diffuse_bio/miniconda/envs/core/lib/python3.9/site-packages/jax
JAXLIB_PATH ?= /home/bryan_diffuse_bio/miniconda/envs/core/lib/python3.9/site-packages/jaxlib

CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc

# Common flags
CXXFLAGS := -O3 -fPIC -std=c++17
CUDA_FLAGS := -O3 -std=c++17 --compiler-options '-fPIC -std=c++17'
CUDA_ARCH := -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75

# Include paths
INCLUDES := -I$(CUDA_PATH)/include \
           -I$(PYTHON_PATH) \
           -I$(PYBIND11_PATH) \
           -I. \
           -I$(JAX_PATH) \
           -I$(JAXLIB_PATH) \
           -I$(JAXLIB_PATH)/include \
           -I$(JAXLIB_PATH)/xla_extension \
           -I$(JAXLIB_PATH)/_src \
           -I$(JAXLIB_PATH)/_src/xla/pjrt/c \
           -I$(JAXLIB_PATH)/_src/xla/service

# Source files
CUDA_SRC := triangle_attention.cu
CPP_SRC := triangle_attention_kernel.cc

# Output
TARGET := libtriangle_attention.so

# Rules
all: $(TARGET)

$(TARGET): $(CUDA_SRC) $(CPP_SRC)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) $(INCLUDES) -shared $^ -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean 