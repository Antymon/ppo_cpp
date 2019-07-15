//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_CART_POLE_ENV_HPP
#define PPO_CPP_CART_POLE_ENV_HPP

#include <string>
#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class CartPoleEnv : public virtual Env{
public:
    CartPoleEnv(int num_envs):Env(num_envs){

    }

    std::string get_action_space() override {
        return Env::SPACE_CONTINOUS;
    }

    std::string get_observation_space() override {
        return Env::SPACE_CONTINOUS;
    }

    int get_action_space_size() override{
        return 2;
    }

    int get_observation_space_size() override{
        return 4;
    }

};

#endif //PPO_CPP_CART_POLE_ENV_HPP
