
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcufft -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:		cuRAND

cuRAND:	cuRAND.cu
		nvcc cuRAND.cu -o cuRAND $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f cuRAND
