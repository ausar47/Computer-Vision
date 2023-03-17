cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)
project(transform)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp example_18-01_from_disk.cpp example_19-01.cpp hw5.h)
add_executable(HW5 ${SOURCE_FILES})
target_link_libraries (HW5 ${OpenCV_LIBS})
