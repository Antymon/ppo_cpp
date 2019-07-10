cmake_minimum_required(VERSION 3.14)
project(ppo_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories("/usr/local/include/eigen3")
include_directories("/usr/local/include/tensorflow")
include_directories("/usr/local/include/protobuf")

add_executable(test main.cpp network_loader.hpp)
add_executable(ppo_cpp ppo2.cpp ppo2.hpp ActorCriticRLModel.hpp policies.hpp env.hpp logger.hpp)

target_link_libraries(ppo_cpp "/usr/local/lib/libprotobuf.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libprotoc.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libtensorflow_cc.so")
target_link_libraries(ppo_cpp "/usr/local/lib/libtensorflow_framework.so")
