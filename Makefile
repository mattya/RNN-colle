# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
export CFLAGS = -g -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I./mshadow/ -fopenmp

LDFLAGS= -lm -lz
ifeq ($(mkl),1)
 LDFLAGS+= -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread
 CFLAGS+= -DMSHADOW_USE_MKL=1 -DMSHADOW_USE_CBLAS=0
else
 LDFLAGS+= -lcblas
 CFLAGS+= -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CBLAS=1
endif

ifeq ($(cuda),1)
 LDFLAGS+= -lcudart -lcublas -lcurand
 CFLAGS+= -DMSHADOW_USE_CUDA=1
else
 CFLAGS+= -DMSHADOW_USE_CUDA=0
 bin/train: src/train.cpp
endif

export NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX)

# specify tensor path
BIN = bin/train
OBJ =
CUOBJ =
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)


$(BIN) :
	$(CXX) $(CFLAGS)  -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu %.cpp, $^)
$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~


