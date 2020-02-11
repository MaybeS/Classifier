cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TORCH_BASE_PATH "/home/jiun/applications/libtorch")
set(OPENCV_BASE_PATH "/home/jiun/applications/opencv-4.2.0/build")

set(CMAKE_PREFIX_PATH "${TORCH_BASE_PATH}")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} pthread)
target_link_libraries(${PROJECT_NAME} stdc++fs)
