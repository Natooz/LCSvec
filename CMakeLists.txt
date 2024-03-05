cmake_minimum_required(VERSION 3.20)
project(lctorch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15)

option(BUILD_PY "Build the python package" ON)
option(BUILD_TEST "Build the test target" OFF)
option(VTUNE "Compile Options for Intel VTune" Off)

# set /utf-8 for MSVC
if(MSVC)
    add_compile_options(-bigobj)
    add_compile_options(/utf-8)
endif()

# include_directories(include)

if(BUILD_PY)
    message("Building python binding.")
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    # set target to O3 and release mode because nanobind use minisize
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
        find_package(nanobind CONFIG REQUIRED)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(my_ext src/main.cpp)
    # target_link_libraries(my_ext PRIVATE lctorch)
    # install(TARGETS my_ext LIBRARY DESTINATION lctorch)
endif()

# optimized release build
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# enable parallel compilation for MSVC
if(MSVC)
    message(STATUS "MSVC parallel compilation enabled")
    add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/Gm->)
    cmake_host_system_information(RESULT CPU_NUMBER_OF_LOGICAL_CORES QUERY NUMBER_OF_LOGICAL_CORES)
    add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP${CPU_NUMBER_OF_LOGICAL_CORES}>)
endif()

if(VTUNE)
    message(STATUS "VTune enabled")
    if(MSVC)
        add_compile_options(/Zi)
        add_compile_options(/MDd)
        add_compile_options(/D "TBB_USE_THREADING_TOOLS")
    else()
        add_compile_options(-g)
    endif ()
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)