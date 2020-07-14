#---------------------------------------------------
# OpenMM Velocity Verlet Plugin
#
# Creates OpenMMVelocityVerlet plugin library.
#
# Windows:
#   OpenMMVelocityVerlet.dll
#   OpenMMVelocityVerlet.lib
# Unix:
#   libOpenMMVelocityVerlet.so
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(OpenMM-VelocityVerlet)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# Specify the C++ version we are building for.
SET(CMAKE_CXX_STANDARD 11)

# Set flags for linking on mac
IF (APPLE)
    SET(CMAKE_INSTALL_NAME_DIR "@rpath")
    SET(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
ENDIF (APPLE)

# Set building flags
IF (UNIX AND NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Debug or Release build" FORCE)
ENDIF (UNIX AND NOT CMAKE_BUILD_TYPE)

IF (NOT CMAKE_CXX_FLAGS_DEBUG)
    SET(CMAKE_CXX_FLAGS_DEBUG "-g" CACHE STRING "To use when CMAKE_BUILD_TYPE=Debug" FORCE)
ENDIF (NOT CMAKE_CXX_FLAGS_DEBUG)

IF (NOT CMAKE_CXX_FLAGS_RELEASE)
    IF (ANDROID)
        SET(CMAKE_CXX_FLAGS_RELEASE "-mfloat-abi=softfp -march=armv7-a -mfpu=neon -funsafe-math-optimizations -O3 -DNDEBUG" CACHE STRING
                "To use when CMAKE_BUILD_TYPE=Release" FORCE)
    ELSE (ANDROID)
        SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING
                "To use when CMAKE_BUILD_TYPE=Release" FORCE)
    ENDIF (ANDROID)
ENDIF (NOT CMAKE_CXX_FLAGS_RELEASE)

# Select where to install
IF (${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF (WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE (WIN32)
        SET(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF (WIN32)
ENDIF (${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
#SET(VELOCITYVERLET_PLUGIN_SOURCE_SUBDIRS openmmapi serialization)
SET(VELOCITYVERLET_PLUGIN_SOURCE_SUBDIRS openmmapi)

# Set the library name
SET(VELOCITYVERLET_LIBRARY_NAME OpenMMVelocityVerlet)
SET(SHARED_VELOCITYVERLET_TARGET ${VELOCITYVERLET_LIBRARY_NAME})

# These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/openmm")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH (dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH (dir)

# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH (subdir ${VELOCITYVERLET_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES ${SOURCE_FILES} ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH (subdir)

# Create the library.

ADD_LIBRARY(${SHARED_VELOCITYVERLET_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_VELOCITYVERLET_TARGET}
        PROPERTIES COMPILE_FLAGS "-DVELOCITYVERLET_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
        LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_VELOCITYVERLET_TARGET} OpenMM OpenMMDrude)
INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_VELOCITYVERLET_TARGET})

# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/openmm/internal/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Enable testing

ENABLE_TESTING()
#ADD_SUBDIRECTORY(serialization/tests)

# Build the implementations for different platforms

#ADD_SUBDIRECTORY(platforms/reference)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
FIND_PACKAGE(OpenCL QUIET)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
FIND_PACKAGE(OpenCL QUIET)
#IF(OPENCL_FOUND)
#    SET(VELOCITYVERLET_BUILD_OPENCL_LIB ON CACHE BOOL "Build implementation for OpenCL")
#ELSE(OPENCL_FOUND)
SET(VELOCITYVERLET_BUILD_OPENCL_LIB OFF CACHE BOOL "Build implementation for OpenCL")
#ENDIF(OPENCL_FOUND)
IF (VELOCITYVERLET_BUILD_OPENCL_LIB)
    ADD_SUBDIRECTORY(platforms/opencl)
ENDIF (VELOCITYVERLET_BUILD_OPENCL_LIB)

FIND_PACKAGE(CUDA QUIET)
IF (CUDA_FOUND)
    SET(VELOCITYVERLET_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA")
ELSE (CUDA_FOUND)
    SET(VELOCITYVERLET_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA")
ENDIF (CUDA_FOUND)
IF (VELOCITYVERLET_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF (VELOCITYVERLET_BUILD_CUDA_LIB)

# Build the Python API

FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
IF (PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(VELOCITYVERLET_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE (PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(VELOCITYVERLET_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF (PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
IF (VELOCITYVERLET_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF (VELOCITYVERLET_BUILD_PYTHON_WRAPPERS)

# Fix windows compilation
IF (WIN32)
    ADD_COMPILE_DEFINITIONS(OPENMM_VELOCITYVERLET_BUILDING_SHARED_LIBRARY)
ENDIF ()
