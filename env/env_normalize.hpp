//
// Created by szymon on 17/07/19.
//

#ifndef PPO_CPP_ENV_NORMALIZE_HPP
#define PPO_CPP_ENV_NORMALIZE_HPP

#include "env.hpp"
#include "../common/running_statistics.hpp"
#include "hexapod_env.hpp"
#include "../common/matrix_clamp.hpp"
#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::RowVectorXf RowVector;

class EnvNormalize : public virtual Env {
public:
    explicit EnvNormalize(
            Env &env,
            bool norm_obs = true,
            bool norm_reward = true,
            float clip_reward = 10,
            float clip_obs = 10,
            float gamma = 0.99,
            float epsilon = 1e-8)
            : Env(env.get_num_envs()),
            env{env},
            norm_obs{norm_obs},
            norm_reward{norm_reward},
            ret{Mat::Zero(env.get_num_envs(),1)},
            gamma{gamma},
            epsilon{epsilon},
            obs_rms{RunningStatistics(env.get_num_envs())},
            ret_rms{},
            clamp_obs{env.get_num_envs(),env.get_observation_space_size(),clip_obs},
            clamp_rewards{ret,clip_reward},
            ret_like_ones{Mat::Ones(ret.rows(),ret.cols())}
    {}

    std::string get_action_space() override {
        return env.get_action_space();
    }

    std::string get_observation_space() override {
        return env.get_observation_space();
    }

    int get_action_space_size() override {
        return env.get_action_space_size();
    }

    int get_observation_space_size() override {
        return env.get_observation_space_size();
    }

    int get_num_envs() override {
        return env.get_num_envs();
    }

    std::vector<Mat> step(const Mat &actions) override {
        const std::vector<Mat>& results = env.step(actions);

        Mat rews {results[1]};

//        std::cout << rews <<std::endl;

        ret = ret * gamma + rews;
//        std::cout << ret  <<std::endl;

        Mat obs{_normalize_observation(results[0])};
        if (norm_reward){
            ret_rms.update(ret);
            rews *= (ret_rms.var + epsilon * RowVector::Ones(rews.cols())).cwiseSqrt().cwiseInverse().asDiagonal();
//            std::cout << rews <<std::endl;
            rews = clamp_rewards.clamp(rews);
//            std::cout << rews <<std::endl;

        }

        //reward for terminal state will be set to 0
        ret = ret.cwiseProduct(ret_like_ones-results[2]);
//        std::cout << ret <<std::endl;

        return {std::move(obs),std::move(rews),results[2]};
    }

    Mat _normalize_observation(const Mat& obs){
        if (norm_obs) {
            obs_rms.update(obs);

            Mat o{(obs.rowwise() - obs_rms.mean) *
                  (obs_rms.var + epsilon * RowVector::Ones(obs.cols())).cwiseSqrt().cwiseInverse().asDiagonal()};

            return clamp_obs.clamp(o);
        }
        else {
            return obs;
        }
    }

    Mat reset() override {
        const Mat& obs = env.reset();

        ret = Mat::Zero(get_num_envs(),1);
        return _normalize_observation(obs);
    }

    void render() override {
        env.render();
    }

    float get_time() override {
        return env.get_time();
    }

    Mat get_original_obs(){
        return env.get_original_obs();
    }

    Mat get_original_rew(){
        return env.get_original_rew();
    }

private:
    Env& env;
    bool norm_obs;
    bool norm_reward;
    Mat ret;
    float gamma;
    float epsilon;
    RunningStatistics obs_rms;
    RunningStatistics ret_rms;
    MatrixClamp clamp_obs;
    MatrixClamp clamp_rewards;
    const Mat ret_like_ones;
};


#endif //PPO_CPP_ENV_NORMALIZE_HPP
