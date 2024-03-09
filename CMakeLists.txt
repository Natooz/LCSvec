cmake_minimum_required(VERSION 3.20)
project(lcsvec LANGUAGES CXX)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

# Find Python and nanobind
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

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
nanobind_add_module(
    # Extension name
    lcsvec_ext

    # Target the stable ABI for Python 3.12+, which reduces
    # the number of binary wheels that must be built. This
    # does nothing on older Python versions
    STABLE_ABI

    # Source code
    src/lcs.cpp)
# target_link_libraries(lcsvec PRIVATE lcsvec)
install(TARGETS lcsvec_ext LIBRARY DESTINATION lcsvec)
