cmake_minimum_required (VERSION 3.9)

project (Animations)
set (CMAKE_VERBOSE_MAKEFILE OFF) # this can be changed with `make VERBOSE=1`

# this is enough to use CORSIKA8
find_package (corsika CONFIG REQUIRED)

add_executable (shower shower.cpp)
target_link_libraries (shower CORSIKA8::CORSIKA8)
