CXX = nvcc
CXXFLAGS = -I ./include,/share/apps/boost/include,$(HOME)/include
LDFLAGS = -L /share/apps/boost/lib,$(HOME)/lib,$(HOME)/lib,/usr/local/cuda/extras/CUPTI/lib64 -lcupti -lglut -lboost_system

COMPILE.cu = $(COMPILE.cc)

%.o: %.cu
	@echo -e "\tNVC\t$@";
	@$(COMPILE.cu) $(OUTPUT_OPTION) $<

%.o: %.cpp
	@echo -e "\tGXX\t$@";
	@$(COMPILE.cpp) $(OUTPUT_OPTION) $<

EXECS = pathtracer

.PHONY: default clean

default: pathtracer

all: $(EXECS)

pathtracer:\
	main.o \
	displayfunc.o \
	kernel.o \
	geometry.o \
	rply.o \
	bvhaccel.o \
	transform.o \
	matrix4x4.o \
	bbox.o
	@echo -e "\tLNK\t$@";
	@$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) $^ $(LDFLAGS) $(LOADLIBES) $(LDLIBS) $(OUTPUT_OPTION)

kernel.o: CXXFLAGS+=-arch=sm_20

clean:
	$(RM) $(EXECS)
	$(RM) *.o
	
bbox.o:\
	scene/bbox.cpp include/geometry/bbox.h include/geometry/vector.h \
	include/utils.h include/pathtracer.h include/config.h \
	include/pathtracer.h include/geometry/point.h include/ray.h \
	include/geometry/vector.h include/geometry/point.h
	@echo -e "\tGXX\t$@";
	@$(COMPILE.cpp) $(OUTPUT_OPTION) $<

bvhaccel.o:\
	scene/bvhaccel.cpp include/geometry/bvhaccel.h \
	include/geometry/vector.h include/utils.h include/pathtracer.h \
	include/config.h include/pathtracer.h include/geometry/point.h \
	include/geometry/triangle.h include/geometry/normal.h \
	include/geometry/vector_normal.h include/geometry/spectrum.h \
	include/geometry/bbox.h include/ray.h include/geometry/vector.h \
	include/geometry/point.h
	@echo -e "\tGXX\t$@";
	@$(COMPILE.cpp) $(OUTPUT_OPTION) $<

geometry.o:\
	scene/geometry.cpp include/geometry.h \
 	include/geometry/point.h include/geometry/vector.h include/utils.h \
 	include/pathtracer.h include/config.h include/pathtracer.h \
 	include/geometry/normal.h include/geometry/vector_normal.h \
	include/geometry/spectrum.h include/geometry/triangle.h \
	include/geometry/point.h include/geometry/normal.h \
	include/geometry/spectrum.h include/geometry/bbox.h include/ray.h \
	include/geometry/vector.h include/geometry/rply.h \
	include/geometry/light.h include/utils.h
	@echo -e "\tGXX\t$@";
	@$(COMPILE.cpp) $(OUTPUT_OPTION) $<
	
matrix4x4.o:\
	scene/matrix4x4.cpp include/geometry/matrix4x4.h
	@echo -e "\tGXX\t$@";
	@$(COMPILE.cpp) $(OUTPUT_OPTION) $<

rply.o:\
	scene/rply.cpp include/geometry/rply.h
	@echo -e "\tGXX\t$@";
	@$(COMPILE.cpp) $(OUTPUT_OPTION) $<

transform.o:\
	scene/transform.cpp include/utils.h include/pathtracer.h \
	include/config.h include/geometry/transform.h include/geometry/point.h \
	include/geometry/vector.h include/pathtracer.h include/geometry/normal.h \
	include/geometry/vector_normal.h include/ray.h include/geometry/vector.h \
	include/geometry/point.h include/geometry/bbox.h \
	include/geometry/matrix4x4.h
	@echo -e "\tGXX\t$@";
	@$(COMPILE.cpp) $(OUTPUT_OPTION) $<
