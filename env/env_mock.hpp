//
// Created by szymon on 10/07/19.
//
//quasi OpenAI gym env interface
//

#ifndef PPO_CPP_MOCK_ENV_HPP
#define PPO_CPP_MOCK_ENV_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>

#include "env.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class EnvMock : public virtual Env
{
public:

    EnvMock(double scaling_coeff = 0.)
        : Env(1)
        , total_step{0}
        , scaling_coeff{scaling_coeff} {
    }

    std::string get_action_space() override {
        return Env::SPACE_CONTINOUS;
    }

    std::string get_observation_space() override {
        return Env::SPACE_CONTINOUS;
    }

    int get_action_space_size() override{
        return 18;
    }

    int get_observation_space_size() override{
        return 18;
    }

    Mat reset() override {
        return scaling_coeff*Mat::Ones(get_num_envs(),get_observation_space_size());
    }

    std::vector<Mat> step(const Mat &actions) override {
        ++total_step;

        auto obs = scaling_coeff*Mat::Ones(get_num_envs(),get_observation_space_size());
        auto rewards = scaling_coeff*Mat::Ones(get_num_envs(), 1);
        Mat dones;

        if(total_step%300==0){
            dones = Mat::Ones(get_num_envs(), 1);
        } else{
            dones = Mat::Zero(get_num_envs(), 1);
        }
        return {obs,rewards,dones};
    }

    Mat get_original_obs(){
        auto obs = scaling_coeff*Mat::Ones(get_num_envs(),get_observation_space_size());
        return std::move(obs);
    }

    Mat get_original_rew(){
        auto rewards = scaling_coeff*Mat::Ones(get_num_envs(), 1);
        return std::move(rewards);
    }
    void serialize(nlohmann::json& json) override {

    }

    void deserialize(nlohmann::json& json) override {

    }

    void render() override {

    }

    float get_time() override {
        return 0;
    }

private:
    long total_step;
    double scaling_coeff;


};

#endif //PPO_CPP_MOCK_ENV_HPP
