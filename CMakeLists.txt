cmake_minimum_required(VERSION 3.14)
project(ppo_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories("/usr/local/include/eigen3")
include_directories("/usr/local/include/eigen3/unsuppored")
include_directories("/usr/local/include/tensorflow")
include_directories("/usr/local/include/protobuf")
include_directories("/ppo2")
include_directories("/env")
include_directories("/usr/local/include") #dart
include_directories("/home/szymon/robot_dart/src") #robot_dart

add_executable(test test/main.cpp test/network_loader.hpp)
add_executable(ppo_cpp ppo2.cpp env/hexapod_env.hpp)
add_executable(test2 test/main2.cpp)
add_executable(test3 test/main3.cpp)
add_executable(test4 test/main4.cpp)

target_link_libraries(ppo_cpp "/usr/local/lib/libprotobuf.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libprotoc.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libtensorflow_cc.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libtensorflow_framework.so")
