#!/bin/bash

./waf configure --exp ppo_cpp --dart /workspace --kdtree /workspace/include --cpp14=yes --pthread=yes
./waf --exp ppo_cpp
