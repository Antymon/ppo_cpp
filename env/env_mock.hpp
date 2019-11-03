//
// Created by szymon on 10/07/19.
//
//quasi OpenAI gym env interface
//

#ifndef PPO_CPP_CART_POLE_ENV_HPP
#define PPO_CPP_CART_POLE_ENV_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class EnvMock : public Env{
public:
    EnvMock(int num_envs):Env(num_envs), total_step{0}{

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

    Mat get_original_obs(){
        auto obs = Mat::Zero(get_num_envs(),get_observation_space_size());
        return std::move(obs);
    }

    Mat get_original_rew(){
        auto rewards = Mat::Zero(get_num_envs(), 1);
        return std::move(rewards);
    }
    void serialize(nlohmann::json& json) override {

    }

    void deserialize(nlohmann::json& json) override {

    }

private:
    long total_step;

};

#endif //PPO_CPP_CART_POLE_ENV_HPP
