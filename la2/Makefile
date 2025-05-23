# Adapted from Job Vranish (https://spin.atomicobject.com/2016/08/26/makefile-c-projects/)
TARGET_EXEC := la2
# DEFAULT_RUN_CONFIG := -t ./dat/train_xor.dat -T ./dat/test_xor.dat -i 1000 -l 1 -h 16 -e 0.7 -m 1 -f 1 -s
# DEFAULT_RUN_CONFIG := -t ./dat/train_nomnist.dat -T ./dat/test_nomnist.dat -i 500 -l 1 -h 4 -f 1 -s
# DEFAULT_RUN_CONFIG := -t ./dat/train_nomnist.dat -T ./dat/test_nomnist.dat -i 500 -l 1 -h 4 -e 1 -m 2 -f 1 -s
# DEFAULT_RUN_CONFIG := -t ./dat/train_nomnist.dat -T ./dat/test_nomnist.dat -i 500 -l 1 -h 4 -e 1 -m 2 -f 0 -s
# DEFAULT_RUN_CONFIG := -t ./dat/train_nomnist.dat -T ./dat/test_nomnist.dat -i 500 -l 1 -h 8 -e 0.1 -m 2 -f 1 -s -o
# DEFAULT_RUN_CONFIG := -t ./dat/train_compas.dat -T ./dat/test_compas.dat

# Mejor arquitectura de la p1
# DEFAULT_RUN_CONFIG := -t ./dat/train_xor.dat -T ./dat/test_xor.dat -i 1000 -l 2 -h 100 -e 0.7 -m 1 -f 1 -s
# DEFAULT_RUN_CONFIG := -t ./dat/train_xor.dat -T ./dat/test_xor.dat -i 1000 -l 2 -h 100 -e 0.7 -m 1 -f 1 -s

# DEFAULT_RUN_CONFIG := -t ./dat/train_compas.dat -T ./dat/test_compas.dat -i 1000 -l 2 -h 8 -e 0.7 -m 1 -f 1 -s

DEFAULT_RUN_CONFIG := -t ./dat/train_nomnist.dat -T ./dat/test_nomnist.dat -i 500 -l 1 -h 8 -e 0.7 -m 1 -f 1 -s

BUILD_DIR := ./bin
SRC_DIRS := .

# Find all the C and C++ files we want to compile
# Note the single quotes around the * expressions. The shell will incorrectly expand these otherwise, but we want to send the * directly to the find command.
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')

# Prepends BUILD_DIR and appends .o to every src file
# As an example, ./your_dir/hello.cpp turns into ./build/./your_dir/hello.cpp.o
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# String substitution (suffix version without %).
# As an example, ./build/hello.cpp.o turns into ./build/hello.cpp.d
DEPS := $(OBJS:.o=.d)

# Paths to OpenCV
OPENCV_DIR := /usr/include/opencv4
OPENCV_INCLUDE_DIR := $(OPENCV_DIR)
OPENCV_LIB_DIR := /usr/lib/x86_64-linux-gnu

# Include flags
# INC_FLAGS := -I.
INC_FLAGS := -I$(OPENCV_INCLUDE_DIR) -I.
# Removed automatic includes:
# # Every folder in ./src will need to be passed to GCC so that it can find header files
# INC_DIRS := $(shell find $(SRC_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
# INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# Linking flags
LDFLAGS := -L$(OPENCV_LIB_DIR) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS := $(INC_FLAGS) -MMD -MP -O3

# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C source
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
    
.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
-include $(DEPS)


run: $(BUILD_DIR)/$(TARGET_EXEC)
	$(BUILD_DIR)/$(TARGET_EXEC) $(DEFAULT_RUN_CONFIG)


