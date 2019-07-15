//
// Created by szymon on 10/07/19.
//

#include "ppo2/ppo2.hpp"
#include "env/env.hpp"
#include "env/cart_pole_v1_env_mock.hpp"


int main(){

    CartPoleEnv e {1};
    PPO2 algorithm {"./exp/ppo_cpp/ppo2_graph_forced_cont_actions.meta.txt",e,
                    0.99,2048,0,1e-3,0.5f,.5,.95,32,10,0.2,-1,"exp/ppo_cpp/tensorboard/"
    };

    algorithm.learn(2e7);

    return 0;
}
