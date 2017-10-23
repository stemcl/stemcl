cmake_minimum_required(VERSION 3.5)
set(MACOSX_DEPLOYMENT_TARGET "macosx10.12")

# Find required libraries
find_package(OpenCL REQUIRED)
find_package(clFFT REQUIRED)

# Library include directories
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CLFFT_INCLUDE_DIRS})
set(LIBS ${LIBS} ${CLFFT_LIBRARIES} ${OpenCL_LIBRARY})

add_executable(stemcl stemcl.c helpers.c)
target_link_libraries(stemcl ${LIBS})

add_custom_command(TARGET stemcl POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:stemcl>/assets)
# Installation
install(TARGETS stemcl RUNTIME DESTINATION bin)
install(FILES assets/* RUNTIME DESTINATION bin)
