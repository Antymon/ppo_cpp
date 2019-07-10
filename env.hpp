//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_ENV_HPP
#define PPO_CPP_ENV_HPP

#include "logger.hpp"
#include <string>

class Env {
public:
    Env(int num_envs):
        _num_envs{num_envs} {

    }

    int get_num_envs() {
        return _num_envs;
    }

private:
    int _num_envs;
};

#endif //PPO_CPP_ENV_HPP
