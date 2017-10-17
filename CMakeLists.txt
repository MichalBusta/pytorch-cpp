cmake_minimum_required(VERSION 3.0)
project(pytorch-cpp)

# ATen
set(ATen_INSTALL_DIR  D:/MSVC-build/ATen/install)

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

OPTION (BUILD_SHARED_LIBS "Build Shared Libraries" ON)

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA
find_package(CUDA 5.5)
include_directories(${CUDA_INCLUDE_DIRS})

# HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
include_directories(${HDF5_INCLUDE_DIRS})
# HDF5 need to declare build share lib or not
set(H5_BUILT_AS_DYNAMIC_LIB 1)
set(H5_BUILT_AS_STATIC_LIB 0)
SET(LIB_TYPE SHARED)
SET(H5_ENABLE_SHARED_LIB NO)
SET (H5_ENABLE_STATIC_LIB NO)
if(BUILD_SHARED_LIBS)
  SET (LIB_TYPE SHARED)
  ADD_DEFINITIONS (-DH5_BUILT_AS_DYNAMIC_LIB)
  set(H5_BUILT_AS_DYNAMIC_LIB 1)
  set(H5_BUILT_AS_STATIC_LIB 0)
  SET (H5_ENABLE_SHARED_LIB YES)
else()
  SET (LIB_TYPE STATIC)
  ADD_DEFINITIONS (-DH5_BUILT_AS_STATIC_LIB)
  set(H5_BUILT_AS_DYNAMIC_LIB 0)
  set(H5_BUILT_AS_STATIC_LIB 1)
  SET (H5_ENABLE_STATIC_LIB YES)
endif()

include_directories(
  ${ATen_INSTALL_DIR}/include

  # dense
  ${ATen_INSTALL_DIR}/lib/include/TH
  # ${ATen_INSTALL_DIR}/lib/include/TH/generic
  # ${ATen_INSTALL_DIR}/lib/include/TH/vector
  ${ATen_INSTALL_DIR}/lib/include/THC
  # ${ATen_INSTALL_DIR}/lib/include/THC/generic

  # sparse
  ${ATen_INSTALL_DIR}/lib/include/THS
  # ${ATen_INSTALL_DIR}/lib/include/THS/generic
  ${ATen_INSTALL_DIR}/lib/include/THCS
  # ${ATen_INSTALL_DIR}/lib/include/THCS/generic

  # NN
  ${ATen_INSTALL_DIR}/lib/include/THNN
  # ${ATen_INSTALL_DIR}/lib/include/THNN/generic
  ${ATen_INSTALL_DIR}/lib/include/THCUNN
  # ${ATen_INSTALL_DIR}/lib/include/THCUNN/generic
  )

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)

file(GLOB_RECURSE ATEN_LIBS ${ATen_INSTALL_DIR}/lib/*.lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

message(STATUS ${HDF5_LIBRARIES})
message(STATUS ${HDF5_HL_LIBRARIES})

if(BUILD_SHARED_LIBS)
	add_library(pytorch SHARED ${SRC} ${HDR})
else()
	add_library(pytorch STATIC ${SRC} ${HDR})
endif()

target_link_libraries(pytorch ${ATEN_LIBS} ${HDF5_HL_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES})