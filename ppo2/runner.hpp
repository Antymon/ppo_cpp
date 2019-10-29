//
// Created by szymon on 29/10/2019.
//

#ifndef PPO_CPP_RUNNER_HPP
#define PPO_CPP_RUNNER_HPP

#include <utility>
#include <memory>
#include <string>

#include <Eigen/Dense>

#include "../env/env.hpp"
#include "policies.hpp"
#include "utils.hpp"
#include "tensorboard.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

struct MiniBatch {
    std::shared_ptr<Mat> obs;
    std::shared_ptr<Mat> returns;
    std::shared_ptr<Mat> dones;
    std::shared_ptr<Mat> actions;
    std::shared_ptr<Mat> values;
    std::shared_ptr<Mat> neglogpacs;
    std::shared_ptr<Mat> true_rewards;
    std::shared_ptr<Mat> unnormalized_rewards;

    std::vector<std::shared_ptr<Mat>> get_train_input() const {
        return {obs, returns, dones, actions, values, neglogpacs};
    }

    std::vector<std::shared_ptr<Mat>> get_1_dims() const {
        return {returns, dones, values, neglogpacs,true_rewards, unnormalized_rewards};
    }
};

class Runner {
public:
    Runner(Env& env, MlpPolicy& model, int n_steps, float gamma, float lam)
            : env{env}
            , model{model}
            , n_steps{n_steps}
            , gamma {gamma}
            , lam{lam}
            , obs{env.reset()}
            , num_envs{env.get_num_envs()}
            , dones{Mat::Zero(num_envs,1)}
    {


    }

    MiniBatch run() {
        MiniBatch mb {};

        //srarts with num_envs!
        mb.obs = std::make_shared<Mat>(num_envs,n_steps*env.get_observation_space_size());
        mb.actions = std::make_shared<Mat>(num_envs,n_steps*env.get_action_space_size());

        //starts with n_steps!
        mb.returns = std::make_shared<Mat>(n_steps, num_envs);
        mb.dones = std::make_shared<Mat>(n_steps, num_envs);
        mb.values = std::make_shared<Mat>(n_steps, num_envs);
        mb.neglogpacs = std::make_shared<Mat>(n_steps, num_envs);
        mb.true_rewards = std::make_shared<Mat>(n_steps, num_envs);
        mb.unnormalized_rewards = std::make_shared<Mat>(n_steps, num_envs);

        for (int step = 0; step< n_steps; ++step){

            //all assigned matrices are assumed to have height of num_envs!

            auto tensor_obs = tensorflow::Tensor();
            Utils::convert_mat(obs,tensor_obs);
            mb.obs->block(0,step*env.get_observation_space_size(),num_envs,env.get_observation_space_size())
                    = obs;

            //action, value, neglogp
            const std::vector<tensorflow::Tensor>& step_result = model.step(tensor_obs);

            //std::cout << "#" << 1<<std::endl;

            auto tensor_actions = step_result[0];

            //std::cout << "#" << 1.51<<std::endl;
            Mat actions;// = Mat(num_envs,env.get_action_space_size());
            Utils::convert_tensor(tensor_actions,actions,env.get_action_space());
            //std::cout << "#" << 1.52<<std::endl;
            assert(actions.rows()==num_envs && actions.cols()==env.get_action_space_size());
            //std::cout << "#" << 1.53<<std::endl;
            mb.actions->block(0,step*env.get_action_space_size(),num_envs,env.get_action_space_size())
                    = actions;

            //std::cout << "#" << 1.5<<std::endl;

            auto tensor_values = step_result[1];
            Mat values;// = Mat(num_envs,1);
            Utils::convert_tensor(tensor_values,values);
            assert(values.rows()==num_envs && values.cols()==1);
            mb.values->block(step,0, 1, num_envs) = values.transpose();

            auto tensor_neglogpacs = step_result[2];
            Mat neglogpacs;// = Mat(num_envs,1);
            Utils::convert_tensor(tensor_neglogpacs,neglogpacs);
            assert(neglogpacs.rows()==num_envs && neglogpacs.cols()==1);
            mb.neglogpacs->block(step,0, 1, num_envs) = neglogpacs.transpose();

            mb.dones->block(step,0, 1, num_envs) = dones.transpose();

            //env should handle action clippin
            //std::cout << "#" << 2<<std::endl;

            //self.obs[:], rewards, self.dones
            const std::vector<Mat>& env_step_result = env.step(actions);

            assert(env_step_result[0].rows() == num_envs && env_step_result[0].cols() == env.get_observation_space_size());
            obs = env_step_result[0];

            //num_envs,1
            assert(env_step_result[2].rows() == num_envs && env_step_result[2].cols() == 1);
            dones  = env_step_result[2];

            //obs shape num_envs,observation_space_size);
            assert(env_step_result[1].rows() == num_envs && env_step_result[1].cols() == 1);
            mb.true_rewards->block(step,0,1, num_envs)=env_step_result[1].transpose();

            mb.unnormalized_rewards->block(step,0,1, num_envs)=env.get_original_rew().transpose();

        }

        //advantage calc
        set_returns(mb);

        //reshaping (num_envs,n_steps*space_size) -> (n_steps*num_envs,space_size) [contiguous samples per environment]

        Eigen::Map<Mat> obs_view(mb.obs->data(), num_envs*n_steps,env.get_observation_space_size());
        *(mb.obs) = obs_view;

        Eigen::Map<Mat> actions_view(mb.actions->data(), num_envs*n_steps,env.get_action_space_size());
        *(mb.actions) = actions_view;

        //reshaping (n_steps,num_envs) -> (n_steps*num_envs) [contiguous samples per environment require tranpose]

        const std::vector<std::shared_ptr<Mat>>& mb_1_dims = mb.get_1_dims();

        for(const std::shared_ptr<Mat>& v : mb_1_dims){
            v -> transposeInPlace();
            Eigen::Map<Mat> view(v->data(),num_envs*n_steps,1);
            *v = view;
        }

        //std::cout << "#" << 4 <<std::endl;

        return mb;
    }

    void set_returns(MiniBatch& mb) const{

        auto tensor_obs = tensorflow::Tensor();
        Utils::convert_mat(obs,tensor_obs);
        auto tensor_last_values = model.value(tensor_obs);
        Mat last_values;// = Mat(num_envs,1);
        Utils::convert_tensor(tensor_last_values,last_values);
        assert(last_values.rows()==num_envs && last_values.cols()==1);

        auto mb_advs = std::make_shared<Mat>(n_steps, num_envs);

        Mat nextnonterminal{1, num_envs};
        Mat nextvalues{1, num_envs};
        Mat delta{1, num_envs};

        Mat last_gae_lam{Mat::Zero(1, num_envs)};

        for (int step = n_steps - 1; step >= 0; --step) {
            if (step == n_steps - 1) {

                nextnonterminal = Mat::Ones(1, num_envs) - dones.transpose();
                nextvalues = last_values.transpose();
            } else {
                nextnonterminal = Mat::Ones(1, num_envs) - mb.dones->row(step + 1);
                nextvalues = mb.values->row(step + 1);
            }
            delta = mb.true_rewards->row(step) + gamma * nextvalues.cwiseProduct(nextnonterminal) -
                    mb.values->row(step);
            mb_advs->row(step) = last_gae_lam =
                    delta + gamma * lam * nextnonterminal.cwiseProduct(last_gae_lam);
        }
        *(mb.returns) = *mb_advs + *(mb.values);
    }

private:
    Env& env;
    MlpPolicy& model;
    int n_steps;
    float gamma;
    float lam;
    Mat obs;
    int num_envs;
    Mat dones;
};

#endif //PPO_CPP_RUNNER_HPP
