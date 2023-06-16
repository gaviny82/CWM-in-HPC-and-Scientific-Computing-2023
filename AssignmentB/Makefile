# Compiler flags
INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcuda -lcurand

GCC = g++
GCC_OPTS =-O3 -Wall -Wextra $(INC)

NVCC = ${CUDA_HOME}/bin/nvcc
NVCCFLAGS = -O3 -arch=sm_35 --ptxas-options=-v -Xcompiler -Wextra -lineinfo

# Porject files
SRC_CUDA = mc_pi_cuda.cu
BIN_CUDA = mc_pi_cuda

SRC_OMP = mc_pi_omp.c
BIN_OMP = mc_pi_omp

# Targets
all: $(BIN_CUDA) $(BIN_OMP)

$(BIN_CUDA): $(SRC_CUDA)
	$(NVCC) -o $(BIN_CUDA) $(SRC_CUDA) $(LIB) $(NVCCFLAGS) 
	
$(BIN_OMP): $(SRC_OMP)
	$(GCC) -o $(BIN_OMP) $(SRC_OMP)

clean:	
	rm -f *.o
	rm -f $(BIN_CUDA) $(BIN_CUDA)
