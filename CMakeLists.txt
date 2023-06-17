cmake_minimum_required(VERSION 3.3)
project(StereoCam)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(OpenCV_DIR [OpenCV.cmake PATH])
set(YOLO "../darknet")

find_package(OpenCV REQUIRED)
add_definitions(-DOPENCV)

include_directories(
	${OpenCV_INCLUDE_DIR}
	${YOLO}/include
)

link_directories(
#	${CMAKE_SOURCE_DIR}/lib
	${YOLO}
)

set(SOURCE_FILES ./src/main.cpp)

add_executable(stereoCam ${SOURCE_FILES})
target_link_libraries(stereoCam
	${OpenCV_LIBRARIES}
	darknet
)

add_executable( calibration
	./src/Calibration.cpp
)
target_link_libraries(calibration
	${OpenCV_LIBRARIES}
)
