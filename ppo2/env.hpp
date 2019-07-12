//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_ENV_HPP
#define PPO_CPP_ENV_HPP

#include <string>

class Env {
public:
    Env(int num_envs):
        _num_envs{num_envs} {

    }

    std::string get_action_space(){
        return SPACE_CONTINOUS;
    }

    std::string get_observation_space(){
        return SPACE_CONTINOUS;
    }

    int get_num_envs() {
        return _num_envs;
    }

private:
    const static std::string SPACE_CONTINOUS;
    const static std::string SPACE_DISCRETE;

    int _num_envs;
};

const std::string Env::SPACE_CONTINOUS = "continous";
const std::string Env::SPACE_DISCRETE = "discrete";

#endif //PPO_CPP_ENV_HPP
