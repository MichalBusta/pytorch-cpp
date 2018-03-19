cmake_minimum_required(VERSION 3.0)
project(pytorch-cpp)

# ATen
set(ATen_INSTALL_DIR  ../ATen/build/install)

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

if(MSVC)
  foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/W3")
            string(REGEX REPLACE "/W3" "/w" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/W3")
	if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endif(MSVC)

OPTION (BUILD_SHARED_LIBS "Build Shared Libraries" ON)

# CUDA
find_package(CUDA 5.5)
include_directories(${CUDA_INCLUDE_DIRS})

# HDF5
set (LIB_TYPE STATIC) # or SHARED
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

find_package (HDF5 NAMES hdf5 COMPONENTS C CXX HL ${SEARCH_TYPE})
#find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
include_directories(${HDF5_INCLUDE_DIR})
# HDF5 need to declare build share lib or not
if(WIN32)
  set(H5_BUILT_AS_DYNAMIC_LIB 0)
  set(H5_BUILT_AS_STATIC_LIB 1)
endif()

include_directories(
  ${ATen_INSTALL_DIR}/include

  # dense
  ${ATen_INSTALL_DIR}/lib/include/TH
  ${ATen_INSTALL_DIR}/lib/include/THC

  # sparse
  ${ATen_INSTALL_DIR}/lib/include/THS
  ${ATen_INSTALL_DIR}/lib/include/THCS

  # NN
  ${ATen_INSTALL_DIR}/lib/include/THNN
  ${ATen_INSTALL_DIR}/lib/include/THCUNN
  )

file(
  GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx)
file(GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)

link_directories(${ATen_INSTALL_DIR}/lib)
if (UNIX)
  set(ATEN_LIBS "ATen -lgomp")
else(UNIX)
  set(ATEN_LIBS "ATen")
endif(UNIX)
#file(GLOB_RECURSE ATEN_LIBS ${ATen_INSTALL_DIR}/lib/*.lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(BUILD_SHARED_LIBS)
	add_library(pytorch SHARED ${SRC} ${HDR})
  include(GenerateExportHeader)
  GENERATE_EXPORT_HEADER(pytorch           
    BASE_NAME pytorch
    EXPORT_MACRO_NAME PYTORCH_EXPORTS
    EXPORT_FILE_NAME pytorch_exports.h
    STATIC_DEFINE SHARED_EXPORTS_BUILT_AS_STATIC)
else()
	add_library(pytorch STATIC ${SRC} ${HDR})
endif()

link_directories(D:/devel/HDF5-1.8.20-win64/lib)

if(MSVC)
  target_link_libraries(pytorch  D:/devel/hdf5-1.8.20/hdf5-1.8.20/build/c++/src/Release/libhdf5_cpp.lib D:/devel/hdf5-1.8.20/hdf5-1.8.20/build/src/Release/libhdf5.lib ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} ${ATEN_LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
else(MSVC)
  target_link_libraries(pytorch ${ATEN_LIBS} ${HDF5_HL_LIBRARIES} ${CUDA_LIBRARIES})
endif(MSVC)


#add_subdirectory(examples)