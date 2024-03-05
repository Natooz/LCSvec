cmake_minimum_required(VERSION 3.20)
project(lctorch)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15)

# optimized release build
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# find nanobind
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

# Build the extensions
nanobind_add_module(lctorch src/main.cpp)
# target_link_libraries(lctorch PRIVATE lctorch)
# install(TARGETS lctorch LIBRARY DESTINATION lctorch)
