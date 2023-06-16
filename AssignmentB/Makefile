# Compiler flags
INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcuda -lcurand

GCC = g++
GCC_OPTS =-O3 -Wall -Wextra $(INC)

NVCC = ${CUDA_HOME}/bin/nvcc
NVCCFLAGS = -O3 -arch=sm_35 --ptxas-options=-v -Xcompiler -Wextra -lineinfo

# Porject files
SRC_FILES = mc_pi_cuda.cu
BIN = mc_pi_cuda

# Targets
all: clean compile

compile: Makefile
	$(NVCC) -o $(BIN) $(SRC_FILES) $(LIB) $(NVCCFLAGS) 

clean:	
	rm -f *.o $(ANALYZE)
