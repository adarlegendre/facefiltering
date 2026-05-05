cmake_minimum_required(VERSION 3.16)
project(face_filter_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(face_filter_cli main.cpp)
target_include_directories(face_filter_cli PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face_filter_cli PRIVATE ${OpenCV_LIBS})
