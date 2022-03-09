set(LIBNAME plugin)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(${LIBNAME})

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${LIBNAME} SHARED plugin.cpp)
target_link_libraries(${LIBNAME} ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET ${LIBNAME} PROPERTY CXX_STANDARD 17)
