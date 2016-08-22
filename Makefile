# The compilers to use
CC = g++
NC = nvcc

# C++ Compiler flags
CXXFLAGS = -c -Wall
CXXDEBUGFLAGS = $(CXXFLAGS) -g

# CUDA Compiler flags
NXXFLAGS = -c
NXXDEBUGFLAGS = $(NXXFLAGS) -g

# Include and Library directories
INC_DIR = -Iinc -I/opt/cuda/include
LIB_DIR = -L/opt/cuda/lib64
LIB = -lfreeimage -lcuda -lcudart

# Separate files based on file endings
CC_SRC = $(wildcard src/*.cc)
CU_SRC = $(wildcard src/*.cu)

# Objects for the individual sources
CC_OBJS = $(addprefix obj/, $(notdir $(CC_SRC:%.cc=%.o)))
CU_OBJS = $(addprefix obj/, $(notdir $(CU_SRC:%.cu=%.o)))

all: run

run: $(CU_OBJS) $(CC_OBJS)
	$(CC) $(LIB_DIR) $(LIB) $(CU_OBJS) $(CC_OBJS) -o bin/run

obj/%.o : src/%.cc
	$(CC) $(INC_DIR) $(CXXFLAGS) -o $@ $^
obj/%.o : src/%.cu
	$(NC) $(INC_DIR) $(NXXFLAGS) -o $@ $^

#debug: cmakelean $(CU_OBJS) $(CC_OBJS)
#	$(CC) $(LIB_DIR) $(LIB) $(CU_OBJS) $(CC_OBJS) -o bin/debug

#obj/%.o : src/%.cu
#	$(NC) $(INC_DIR) $(NXXDEBUGFLAGS) -o $@ $^

#obj/%.o : src/%.cc
#	$(CC) $(INC_DIR) $(CXXDEBUGFLAGS) -o $@ $^

clean:
	rm -f obj/* bin/*
