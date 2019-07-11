//
// Created by szymon on 10/07/19.
//

#include "ppo2.hpp"
#include "env.hpp"

int main(){
    Env e {8};
    PPO2 algorithm {"./exp/ppo_cpp/ppo2_graph-1562761538.345122.meta.txt",e};

    return 0;
}
