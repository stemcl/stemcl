cmake_minimum_required(VERSION 3.5)
set(MACOSX_DEPLOYMENT_TARGET "macosx10.12")

# Find required libraries
find_package(OpenCL REQUIRED)
find_package(clFFT REQUIRED)

# Library include directories
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CLFFT_INCLUDE_DIRS})
set(LIBS ${LIBS} ${CLFFT_LIBRARIES} ${OpenCL_LIBRARY} m)

add_executable(stemcl source/stemcl.c source/helpers.c)
target_link_libraries(stemcl ${LIBS})

# Installation
install(TARGETS stemcl RUNTIME DESTINATION bin)
