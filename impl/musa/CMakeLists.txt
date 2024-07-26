include(FindPackageHandleStandardArgs)
cmake_minimum_required(VERSION 3.14)
project(musa_impl)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
# add_executable(myapp test/test_nothing.cpp)

if(NOT DEFINED ENV{MUSA_PATH})
    set(MUSA_PATH /usr/local/musa)
else()
    set(MUSA_PATH $ENV{MUSA_PATH})
endif()

# set(MUSART_KERNELS_LIBRARY_PATH  /home/torch_musa/torch_musa/)

# musa.h musa_runtime_api.h musa_runtime.h
find_path(MUSA_INCLUDE_DIR NAMES musa.h musa_runtime_api.h musa_runtime.h HINTS ${MUSA_PATH}/include)
MESSAGE("MUSA_INCLUDE_DIR:" ${MUSA_INCLUDE_DIR})
IF (NOT MUSA_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR " Could not find musa.h or musa_runtime_api.h or musa_runtime.h Make sure fullstack is installed, or MUSA_PATH is set properly.")
ENDIF()

find_package(Python3 "3.8" REQUIRED
             COMPONENTS Development Interpreter)
if(Python3_FOUND)
  include_directories(${Python3_INCLUDE_DIRS})
  message("Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Cannot find installed Python3 header file directory")
endif()

# libmccl.so
find_library(MUSA_MCCL_LIBRARY mccl
   HINTS ${MUSA_PATH}/lib 
)
MESSAGE("MUSA_MCCL_LIBRARY:" ${MUSA_MCCL_LIBRARY})
IF (NOT MUSA_MCCL_LIBRARY)
  MESSAGE(FATAL_ERROR " Could not find musa mccl lib. Make sure fullstack is installed, or MUSA_PATH is set properly.")
ENDIF()

# libmusart.so
find_library(MUSART_LIBRARY musart
   HINTS ${MUSA_PATH}/lib 
)
MESSAGE("MUSART_LIBRARY:" ${MUSART_LIBRARY})
IF (NOT MUSART_LIBRARY)
  MESSAGE(FATAL_ERROR " Could not find musa runtime lib. Make sure fullstack is installed, or MUSA_PATH is set properly.")
ENDIF()

# libmusa_python.so
# find_library(musa_python_lb  musa_python
#    HINTS ${MUSART_KERNELS_LIBRARY_PATH}/lib 
# )
# MESSAGE("musa_python:" ${musa_python_lb})
# IF (NOT musa_python_lb)
#   MESSAGE(FATAL_ERROR " Could not find libmusa_python so. Make sure fullstack is installed, or ibmusa_python_PATH is set properly.")
# ENDIF()

set(MUDNN_INCLUDE_SEARCH_PATHS $ENV{MUSA_HOME}/include/)
set(MUDNN_LIB_SEARCH_PATHS $ENV{MUSA_HOME}/lib)

find_path(
  MUDNN_INCLUDE_DIR
  NAMES mudnn.h
  PATHS ${MUDNN_INCLUDE_SEARCH_PATHS}
  NO_DEFAULT_PATH)

find_library(
  MUDNN_LIBRARY
  NAMES mudnn
  PATHS ${MUDNN_LIB_SEARCH_PATHS}
  NO_DEFAULT_PATH)

include_directories(${MUSA_INCLUDE_DIR} 
                    ${MUDNN_INCLUDE_DIR}
                              )                        

add_library(diopi_impl SHARED ${CMAKE_CURRENT_SOURCE_DIR}/functions/basic_op.cpp  
                              ${CMAKE_CURRENT_SOURCE_DIR}/common/common.hpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/common/Handle.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/common/Handle.h
                              ${CMAKE_CURRENT_SOURCE_DIR}/common/Utils.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/common/Utils.h
                              ${CMAKE_CURRENT_SOURCE_DIR}/common/DeviceThreadHandles.h
                              ${CMAKE_CURRENT_SOURCE_DIR}/common/MUSAHooksInterface.cpp     
                              ${CMAKE_CURRENT_SOURCE_DIR}/common/MUSAHooksInterface.h  
                              /opt/conda/envs/py38/lib/python3.8/site-packages/torch/lib/libtorch_python.so    
                              /opt/conda/envs/py38/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so  
                              /opt/conda/envs/py38/lib/python3.8/site-packages/torch/lib/libc10.so    
                              )
target_link_libraries(diopi_impl PUBLIC 
                        ${MUSART_LIBRARY}
                        # ${musa_python_lb}                  
                        ${MUDNN_LIBRARY}
                         )
                         
                         
