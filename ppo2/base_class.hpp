//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_BASE_CLASS_HPP
#define PPO_CPP_BASE_CLASS_HPP

#include "../env/env.hpp"

class BaseRLModel {
public:
    explicit BaseRLModel(Env& env):
    env{env},
    action_space{env.get_action_space()},
    observation_space{env.get_observation_space()},
    n_envs{this->env.get_num_envs()},
    num_timesteps{0}{

    }
protected:
    bool _init_num_timesteps(bool reset_num_timesteps=true){
        if (reset_num_timesteps){
            num_timesteps = 0;
        }

        bool new_tb_log = num_timesteps == 0;
        return new_tb_log;
    }

protected:
    Env &env;
    std::string action_space;
    std::string observation_space;
    int n_envs;
    int num_timesteps;
};

class ActorCriticRLModel : public BaseRLModel {
public:
    ActorCriticRLModel(Env& env) : BaseRLModel(env){
        
    }
};


#endif //PPO_CPP_BASE_CLASS_HPP
