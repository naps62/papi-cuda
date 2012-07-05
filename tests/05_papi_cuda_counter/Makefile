HOSTNAME = $(shell hostname)
UNAME = $(shell uname)
CXX = nvcc

MODE ?= RLS

ifeq ($(HOSTNAME),compute-611-2.local)
CXXFLAGS = -I ./include,/share/apps/boost/include,$(HOME)/include,/usr/local/cuda/extras/CUPTI/include,/usr/local/papi4cuda/include
LDFLAGS = -L /share/apps/boost/lib,$(HOME)/lib,$(HOME)/lib,/usr/local/cuda/extras/CUPTI/lib64,/usr/local/papi4cuda/lib -lcupti -lglut -lboost_system -lpapi
else
ifeq ($(UNAME),Darwin)
CXXFLAGS = -I ./include
LDFLAGS = -L /usr/local/cuda/lib,/usr/local/cuda/extras/CUPTI/lib -lcuda -lcupti -lboost_system-mt -Xlinker -framework,OpenGL,-framework,GLUT -arch=sm_20
endif
endif

ifeq ($(MODE),DBG)
CXXFLAGS += -g -G
$(warning DEBUGGING)
endif

COMPILE.cu = $(COMPILE.cc)

%.o: %.cu
	@echo -e "\tNVC\t$@";
	@$(COMPILE.cu) $(OUTPUT_OPTION) $<

%.o: %.cpp
	@echo -e "\tGXX\t$@";
	@$(COMPILE.cpp) $(OUTPUT_OPTION) $<

EXECS = main

.PHONY: default clean

default: main

all: $(EXECS)

main: main.o
	@echo -e "\tLNK\t$@";
	@$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) *.o
	$(RM) $(EXECS)
