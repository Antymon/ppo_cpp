//
// Created by szymon on 10/07/19.
//
//quasi OpenAI gym env interface
//

#ifndef PPO_CPP_VEC_ENV_HPP
#define PPO_CPP_VEC_ENV_HPP

#include <condition_variable>
#include <mutex>
#include <thread>

#include <string>
#include <vector>

#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class VecEnv : public virtual Env{
public:
    VecEnv(std::vector<std::shared_ptr<Env>>& envs)
        : Env(envs.size())
        , envs{envs}
        , total_step{0}
        , threads{std::vector<std::thread>(envs.size())}
        , condition_vars{std::vector<std::condition_variable>(envs.size())}
        , mutexes{std::vector<std::mutex>(envs.size())}
    {
        assert(envs.size() > 0);

        for (int i = 0; i<envs.size(); ++i) {
            threads[i] = std::thread(&VecEnv::start_env_thread, this, i);
        }
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
    std::vector<std::shared_ptr<Env>>& envs;
    long total_step;
    std::vector<std::thread> threads;
    std::vector<std::condition_variable> condition_vars;
    std::vector<std::mutex> mutexes;

    void start_env_thread(int id){

    }

};

#endif //PPO_CPP_VEC_ENV_HPP