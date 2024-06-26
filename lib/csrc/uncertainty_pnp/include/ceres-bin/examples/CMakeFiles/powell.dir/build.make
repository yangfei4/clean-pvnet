# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /pvnet/lib/csrc/uncertainty_pnp/include/ceres-solver-2.0.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin

# Include any dependencies generated for this target.
include examples/CMakeFiles/powell.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/powell.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/powell.dir/flags.make

examples/CMakeFiles/powell.dir/powell.cc.o: examples/CMakeFiles/powell.dir/flags.make
examples/CMakeFiles/powell.dir/powell.cc.o: /pvnet/lib/csrc/uncertainty_pnp/include/ceres-solver-2.0.0/examples/powell.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/powell.dir/powell.cc.o"
	cd /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/powell.dir/powell.cc.o -c /pvnet/lib/csrc/uncertainty_pnp/include/ceres-solver-2.0.0/examples/powell.cc

examples/CMakeFiles/powell.dir/powell.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/powell.dir/powell.cc.i"
	cd /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvnet/lib/csrc/uncertainty_pnp/include/ceres-solver-2.0.0/examples/powell.cc > CMakeFiles/powell.dir/powell.cc.i

examples/CMakeFiles/powell.dir/powell.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/powell.dir/powell.cc.s"
	cd /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvnet/lib/csrc/uncertainty_pnp/include/ceres-solver-2.0.0/examples/powell.cc -o CMakeFiles/powell.dir/powell.cc.s

examples/CMakeFiles/powell.dir/powell.cc.o.requires:

.PHONY : examples/CMakeFiles/powell.dir/powell.cc.o.requires

examples/CMakeFiles/powell.dir/powell.cc.o.provides: examples/CMakeFiles/powell.dir/powell.cc.o.requires
	$(MAKE) -f examples/CMakeFiles/powell.dir/build.make examples/CMakeFiles/powell.dir/powell.cc.o.provides.build
.PHONY : examples/CMakeFiles/powell.dir/powell.cc.o.provides

examples/CMakeFiles/powell.dir/powell.cc.o.provides.build: examples/CMakeFiles/powell.dir/powell.cc.o


# Object files for target powell
powell_OBJECTS = \
"CMakeFiles/powell.dir/powell.cc.o"

# External object files for target powell
powell_EXTERNAL_OBJECTS =

bin/powell: examples/CMakeFiles/powell.dir/powell.cc.o
bin/powell: examples/CMakeFiles/powell.dir/build.make
bin/powell: lib/libceres.a
bin/powell: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
bin/powell: /usr/lib/x86_64-linux-gnu/libglog.so
bin/powell: /usr/lib/x86_64-linux-gnu/libspqr.so
bin/powell: /usr/lib/x86_64-linux-gnu/libtbb.so
bin/powell: /usr/lib/x86_64-linux-gnu/libcholmod.so
bin/powell: /usr/lib/x86_64-linux-gnu/libccolamd.so
bin/powell: /usr/lib/x86_64-linux-gnu/libcamd.so
bin/powell: /usr/lib/x86_64-linux-gnu/libcolamd.so
bin/powell: /usr/lib/x86_64-linux-gnu/libamd.so
bin/powell: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/powell: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/powell: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/powell: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bin/powell: /usr/lib/x86_64-linux-gnu/librt.so
bin/powell: /usr/lib/x86_64-linux-gnu/libcxsparse.so
bin/powell: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/powell: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/powell: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/powell: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bin/powell: /usr/lib/x86_64-linux-gnu/librt.so
bin/powell: /usr/lib/x86_64-linux-gnu/libcxsparse.so
bin/powell: examples/CMakeFiles/powell.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/powell"
	cd /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/powell.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/powell.dir/build: bin/powell

.PHONY : examples/CMakeFiles/powell.dir/build

examples/CMakeFiles/powell.dir/requires: examples/CMakeFiles/powell.dir/powell.cc.o.requires

.PHONY : examples/CMakeFiles/powell.dir/requires

examples/CMakeFiles/powell.dir/clean:
	cd /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/examples && $(CMAKE_COMMAND) -P CMakeFiles/powell.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/powell.dir/clean

examples/CMakeFiles/powell.dir/depend:
	cd /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /pvnet/lib/csrc/uncertainty_pnp/include/ceres-solver-2.0.0 /pvnet/lib/csrc/uncertainty_pnp/include/ceres-solver-2.0.0/examples /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/examples /pvnet/lib/csrc/uncertainty_pnp/include/ceres-bin/examples/CMakeFiles/powell.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/powell.dir/depend

