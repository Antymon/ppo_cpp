cmake_minimum_required(VERSION 3.14)
project(ppo_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories("/usr/local/include/eigen3")
include_directories("/usr/local/include/tensorflow")
include_directories("/usr/local/include/protobuf")
include_directories("/ppo2")

add_executable(test main.cpp network_loader.hpp)
add_executable(ppo_cpp ppo2.cpp)
add_executable(test2 test2/main2.cpp)

target_link_libraries(ppo_cpp "/usr/local/lib/libprotobuf.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libprotoc.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libtensorflow_cc.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libtensorflow_framework.so")
