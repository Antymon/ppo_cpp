//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_ENV_HPP
#define PPO_CPP_ENV_HPP

#include <string>
#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class Env {
public:
    Env(int num_envs):
        _num_envs{num_envs} {
    }

    virtual std::string get_action_space(){
        return SPACE_CONTINOUS;
    }

    virtual std::string get_observation_space(){
        return SPACE_CONTINOUS;
    }

    virtual int get_action_space_size(){
        return 18;
    }

    virtual int get_observation_space_size(){
        return 1;
    }

    int get_num_envs() {
        return _num_envs;
    }

    Mat reset(){
        return Mat::Zero(get_num_envs(),get_observation_space_size());
    }

    ////self.obs[:], rewards, self.dones
    std::vector<Mat> step(const Mat& actions){

        auto obs = Mat::Zero(get_num_envs(),get_observation_space_size());
        auto rewards = Mat::Zero(get_num_envs(), 1);
        auto dones =  Mat::Zero(get_num_envs(), 1);

        return {obs,rewards,dones};
    }

public:
    const static std::string SPACE_CONTINOUS;
    const static std::string SPACE_DISCRETE;

private:
    int _num_envs;
};

const std::string Env::SPACE_CONTINOUS = "continous";
const std::string Env::SPACE_DISCRETE = "discrete";

#endif //PPO_CPP_ENV_HPP
