# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -I "D:/eigen-3.4.0/eigen-3.4.0" -std=c++11 -Wall -Wextra -Wpedantic -O3 -march=native -mtune=native
LDFLAGS = -lws2_32
# Source files
SRCS = main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable
EXEC = my_program

# Default target
all: $(EXEC)

# Link object files to create executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Header file dependencies
DEPS = $(SRCS:.cpp=.d)

%.d: %.cpp
	$(CXX) -MM $< > $@

-include $(DEPS)

# Clean up build files
clean:
	rm -f $(OBJS) $(EXEC) $(DEPS)

.PHONY: all clean