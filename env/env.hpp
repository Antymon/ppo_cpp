//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_ENV_HPP
#define PPO_CPP_ENV_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class Env {
public:
    Env(int num_envs):
        _num_envs{num_envs} {
    }

    virtual std::string get_action_space() = 0;

    virtual std::string get_observation_space() = 0;

    virtual int get_action_space_size() = 0;

    virtual int get_observation_space_size() = 0;

    int get_num_envs() {
        return _num_envs;
    }

    virtual Mat reset() = 0;

    ////self.obs[:], rewards, self.dones
    virtual std::vector<Mat> step(const Mat& actions) = 0;

public:
    const static std::string SPACE_CONTINOUS;
    const static std::string SPACE_DISCRETE;

private:
    int _num_envs;
};

const std::string Env::SPACE_CONTINOUS = "continous";
const std::string Env::SPACE_DISCRETE = "discrete";

#endif //PPO_CPP_ENV_HPP
