//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_PPO2_HPP
#define PPO_CPP_PPO2_HPP

#include <memory>
#include "Eigen/Dense"
#include <string>
#include "base_class.hpp"
#include "env.hpp"
#include <iostream>
#include "session_creator.hpp"
#include "policies.hpp"
#include "utils.hpp"

const std::string BasePolicy::obs_ph{"input/Ob:0"};
const std::string ActorCriticPolicy::action{"output/_action:0"};
const std::string ActorCriticPolicy::neglogp{"output/_neglogp"};
const std::string ActorCriticPolicy::value_flat{"output/_value_flat"};

struct MiniBatch {
    Eigen::MatrixXf obs;
    Eigen::MatrixXf returns;
    Eigen::MatrixXf dones;
    Eigen::MatrixXf actions;
    Eigen::MatrixXf values;
    Eigen::MatrixXf neglogpacs;
};

class Runner {
public:
    Runner(Env& env) {}

    MiniBatch run() {
        MiniBatch mb;

        return mb;
    }
};

class PPO2 : public ActorCriticRLModel {
public:
    PPO2(std::string model_filename,
            Env &env,
         float gamma = 0.99,
         int n_steps = 128,
         float ent_coef = 0.01,
         float learning_rate = 2.5e-4,
         float vf_coef = 0.5f,
         float max_grad_norm = 0.5,
         float lam = 0.95,
         int nminibatches = 4,
         int noptepochs = 4,
         float cliprange = 0.2,
         std::string tensorboard_log = ""
    ) : ActorCriticRLModel(env)
    , gamma{gamma}
    , n_steps{n_steps}
    , ent_coef{ent_coef}
    , learning_rate{learning_rate}
    , vf_coef{vf_coef}
    , max_grad_norm{max_grad_norm}
    , lam{lam}
    , nminibatches{nminibatches}
    , noptepochs{noptepochs}
    , cliprange{cliprange}
    , tensorboard_log{tensorboard_log}
    {


        n_batch = n_envs * n_steps;

        std::cout << "ppo2 " << std::endl;
        std::cout << "gamma " << gamma << std::endl;

        SessionCreator sc{};
        _session = std::move(sc.load_graph(model_filename));

        if (_session == nullptr || !_session) {
            return;
        }

        act_model = std::make_unique<MlpPolicy>(_session);

    }

    void save(std::string save_path) {

    }

    void learn(int total_timesteps) {

        bool new_tb_log = _init_num_timesteps();

        Runner runner{env};

        const MiniBatch &mb = runner.run();

    }

private:
    void _train_step(float learning_rate,
                     float cliprange,
                     Eigen::RowVectorXf obs,
                     Eigen::MatrixXf returns,
                     Eigen::MatrixXf masks,
                     Eigen::MatrixXf actions,
                     Eigen::MatrixXf values,
                     Eigen::MatrixXf neglogpacs,
                     int update
            //writer
    ) {

    }

    float gamma;
    int n_steps;
    float ent_coef;

    float learning_rate;
    float vf_coef;
    float max_grad_norm;
    float lam;
    int nminibatches;
    int noptepochs;

    float cliprange;
    std::string tensorboard_log;
    int n_batch;

    float cliprange_vf;

    std::shared_ptr<tensorflow::Session> _session;
    std::unique_ptr<MlpPolicy> act_model;


};


#endif //PPO_CPP_PPO2_HPP
