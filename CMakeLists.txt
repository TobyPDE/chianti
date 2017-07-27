# Copyright (C) 2017 Google Inc.
# 
# All rights reserved.
#
# This software may be modified and distributed under the terms of the MIT 
# license.  See the LICENSE file for details.

cmake_minimum_required(VERSION 2.8.12)
project(chianti)

add_compile_options(-std=c++11)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package( OpenCV REQUIRED imgproc 
        opencv_core 
        opencv_highgui 
        opencv_imgproc 
        opencv_imgcodecs)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif()

include_directories(include)
add_library(chianti SHARED
        src/augmentors.cc
        src/iterators.cc 
        src/loaders.cc 
        src/providers.cc)

target_link_libraries(chianti ${OpenCV_LIBS})

install(TARGETS chianti
     RUNTIME DESTINATION bin COMPONENT libraries
     LIBRARY DESTINATION lib COMPONENT libraries
     ARCHIVE DESTINATION lib/static COMPONENT libraries)

add_subdirectory(python)
