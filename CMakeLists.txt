cmake_minimum_required(VERSION 3.3)
project(StereoCam)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(OpenCV_DIR [OpenCV.cmake PATH])

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})

set(SOURCE_FILES ./src/main.cpp)


add_executable(stereoCam ${SOURCE_FILES})

target_link_libraries(stereoCam ${OpenCV_LIBRARIES})
