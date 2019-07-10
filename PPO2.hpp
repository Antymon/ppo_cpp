//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_PPO2_HPP
#define PPO_CPP_PPO2_HPP

#include <memory>
#include "Eigen/Dense"
#include <string>
#include "ActorCriticRLModel.hpp"
#include <iostream>

struct MiniBatch{
    Eigen::MatrixXf obs;
    Eigen::MatrixXf returns;
    Eigen::MatrixXf dones;
    Eigen::MatrixXf actions;
    Eigen::MatrixXf values;
    Eigen::MatrixXf neglogpacs;
};

class Runner {
public:
    Runner() {}

    MiniBatch run(){
        MiniBatch mb;
    }
};

class PPO2 : public virtual ActorCriticRLModel {
public:
    PPO2(float gamma = 0.99,
         int n_steps = 128,
         float ent_coef = 0.01,
         float learning_rate = 2.5e-4,
         float vf_coef = 0.5f,
         float max_grad_norm = 0.5,
         float lam = 0.95,
         int nminibatches = 4,
         int noptepochs = 4,
         float cliprange = 0.2,
         std::string tensorboard_log = "",
         std::string model_filename = "") {

        std::cout << "ppo2 " << std::endl;
        std::cout << "gamma " << gamma<< std::endl;
    }

    void save(std::string save_path){

    }

    void learn(int total_timesteps){
        Runner runner{};

        const MiniBatch& mb = runner.run();

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




};


#endif //PPO_CPP_PPO2_HPP
