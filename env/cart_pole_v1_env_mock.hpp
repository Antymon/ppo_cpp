//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_CART_POLE_ENV_HPP
#define PPO_CPP_CART_POLE_ENV_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class CartPoleEnv : public virtual Env{
public:
    CartPoleEnv(int num_envs):Env(num_envs), total_step{0}{

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

    Mat reset() override {
        return Mat::Zero(get_num_envs(),get_observation_space_size());
    }

    std::vector<Mat> step(const Mat &actions) override {
        ++total_step;

        auto obs = Mat::Zero(get_num_envs(),get_observation_space_size());
        auto rewards = Mat::Zero(get_num_envs(), 1);
        Mat dones;

        if(total_step%300==0){
            dones = Mat::Ones(get_num_envs(), 1);
        } else{
            dones = Mat::Zero(get_num_envs(), 1);
        }
        return {obs,rewards,dones};
    }

private:
    long total_step;

};

#endif //PPO_CPP_CART_POLE_ENV_HPP