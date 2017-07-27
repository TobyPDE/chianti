# Copyright (C) 2017 Google Inc.
# 
# All rights reserved.
#
# This software may be modified and distributed under the terms of the MIT 
# license.  See the LICENSE file for details.

cmake_minimum_required(VERSION 2.8.12)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/python/cmake/Modules)

# The following snippet is provided under the following license and copyright
# ################################### START ####################################
# COPYRIGHT
# 
# Copyright (c) 2014-2017, Anatoly Baksheev
# Copyright (c) 2014-2017, philkr
# Copyright (c) 2014-2017, Cyprien Noel
# Copyright (c) 2014-2017, Ivan Shapovalov
# All rights reserved.
# 
# LICENSE
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set(python_version "2" CACHE STRING "Specify which Python version to use")

if(NOT "${python_version}" VERSION_LESS "3.0.0")
    # Use python3
    find_package(PythonInterp 3)
    find_package(PythonLibs 3)
    find_package(NumPy 1.7.1)

    # Find the matching boost python implementation
    set(version ${PYTHONLIBS_VERSION_STRING})

    STRING( REGEX REPLACE "[^0-9]" "" boost_py_version "${version}" )
    find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
    set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

    while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
        STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

        STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
        find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
        set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

        STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
        if("${has_more_version}" STREQUAL "")
            break()
        endif()
    endwhile()
    if(NOT Boost_PYTHON_FOUND)
        find_package(Boost 1.46 COMPONENTS python)
    endif()
else()
    # Use python 2
    find_package(PythonInterp 2.7)
    find_package(PythonLibs 2.7)
    find_package(NumPy 1.7.1)
    find_package(Boost 1.46 COMPONENTS python)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print(get_python_lib())"
        OUTPUT_VARIABLE PYTHON_PACKAGES_PATH 
        OUTPUT_STRIP_TRAILING_WHITESPACE)

# #################################### END #####################################

if(Boost_FOUND)
    include_directories(
            ../include 
            include
            ${PYTHON_INCLUDE_DIRS}
            ${NUMPY_INCLUDE_DIR}
            PUBLIC 
            ${Boost_INCLUDE_DIRS})

    add_library(pychianti MODULE 
            src/augmentors.cc
            src/iterators.cc 
            src/loaders.cc 
            src/providers.cc 
            src/pychianti.cc)

    target_link_libraries(pychianti 
        chianti 
        ${PYTHON_LIBRARY}
        ${OpenCV_LIBS} 
        ${Boost_LIBRARIES})

    set_target_properties(
            pychianti PROPERTIES
            ARCHIVE_OUTPUT_NAME pychianti
            PREFIX ""
            OUTPUT_NAME pychianti)

    install(TARGETS pychianti
            RUNTIME DESTINATION "${PYTHON_PACKAGES_PATH}" COMPONENT python
            LIBRARY DESTINATION "${PYTHON_PACKAGES_PATH}" COMPONENT python
            ARCHIVE DESTINATION "${PYTHON_PACKAGES_PATH}" COMPONENT python)
endif()
