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
#include "tensorboard.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

const std::string BasePolicy::obs_ph{"input/Ob:0"};
const std::string ActorCriticPolicy::action{"output/_action:0"};
const std::string ActorCriticPolicy::neglogp{"output/_neglogp"};
const std::string ActorCriticPolicy::value_flat{"output/_value_flat"};

struct MiniBatch {
    std::shared_ptr<Mat> obs;
    std::shared_ptr<Mat> returns;
    std::shared_ptr<Mat> dones;
    std::shared_ptr<Mat> actions;
    std::shared_ptr<Mat> values;
    std::shared_ptr<Mat> neglogpacs;
    std::shared_ptr<Mat> true_rewards;

    std::vector<std::shared_ptr<Mat>> get_all() const {
        return {obs, returns, dones, actions, values, neglogpacs, true_rewards};
    }

    std::vector<std::shared_ptr<Mat>> get_1_dims() const {
        return {returns, dones, values, neglogpacs,true_rewards};
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

        //srarts with num envs!
        mb.obs = std::make_shared<Mat>(num_envs,n_steps*env.get_observation_space_size());
        mb.returns = std::make_shared<Mat>(n_steps, num_envs);
        mb.dones = std::make_shared<Mat>(n_steps, num_envs);

        //srarts with num envs!
        mb.actions = std::make_shared<Mat>(num_envs,n_steps*env.get_action_space_size());
        mb.values = std::make_shared<Mat>(n_steps, num_envs);
        mb.neglogpacs = std::make_shared<Mat>(n_steps, num_envs);
        mb.true_rewards = std::make_shared<Mat>(n_steps, num_envs);

        for (int step = 0; step< n_steps; ++step){

            //all assigned matrices are assumed to have height of num_envs!

            auto tensor_obs = tensorflow::Tensor();
            Utils::convert_mat(obs,tensor_obs);
            mb.obs->block(0,step*env.get_observation_space_size(),num_envs,env.get_observation_space_size())
                    = obs;

            //action, value, neglogp
            const std::vector<tensorflow::Tensor>& step_result = model.step(tensor_obs);

            auto tensor_actions = step_result[0];
            assert(tensor_actions.dim_size(0)==num_envs && tensor_actions.dim_size(1)==env.get_action_space_size());
            auto actions = Mat(num_envs,env.get_action_space_size());
            Utils::convert_tensor(tensor_actions,actions);
            mb.actions->block(0,step*env.get_action_space_size(),num_envs,env.get_action_space_size())
                    = actions;

            auto tensor_values = step_result[1];
            assert(tensor_values.dim_size(0)==num_envs && tensor_values.dim_size(1)==1);
            auto values = Mat(num_envs,1);
            Utils::convert_tensor(tensor_values,values);
            mb.values->block(step,0, 1, num_envs) = values.transpose();

            auto tensor_neglogpacs = step_result[2];
            assert(tensor_neglogpacs.dim_size(0)==num_envs && tensor_neglogpacs.dim_size(1)==1);
            auto neglogpacs = Mat(num_envs,1);
            Utils::convert_tensor(tensor_neglogpacs,neglogpacs);
            mb.neglogpacs->block(step,0, 1, num_envs) = neglogpacs.transpose();

            mb.dones->block(step,0, 1, num_envs) = dones.transpose();

            //env should handle action clippin

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
            *v = v->transpose();
            Eigen::Map<Mat> view(v->data(),num_envs*n_steps,1);
            *v = view;
        }

        return mb;
    }

    void set_returns(MiniBatch& mb){

        auto tensor_obs = tensorflow::Tensor();
        Utils::convert_mat(obs,tensor_obs);
        auto tensor_last_values = model.value(tensor_obs);
        auto last_values = Mat(num_envs,1);
        Utils::convert_tensor(tensor_last_values,last_values);

        auto mb_advs = std::make_shared<Mat>(n_steps, num_envs);

        Mat nextnonterminal{1, num_envs};
        Mat nextvalues{1, num_envs};
        Mat delta{1, num_envs};

        Mat last_gae_lam{Mat::Zero(1, num_envs)};

        for (int step = n_steps - 1; step >= 0; --step) {
            if (step == n_steps - 1) {

                nextnonterminal = Mat::Ones(1, num_envs) - dones;
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
    , episode_reward{Mat()}
    {
        n_batch = n_envs * n_steps;

        std::cout << "ppo2 " << std::endl;
        SessionCreator sc{};
        _session = std::move(sc.load_graph(model_filename));

        if (_session == nullptr || !_session) {
            return;
        }

        act_model = std::make_unique<MlpPolicy>(_session);

    }

    void save(std::string save_path) {

    }

    void learn(int total_timesteps, std::string tb_log_name = "PPO2") {

        bool new_tb_log = _init_num_timesteps();


        TensorboardWriter writer {tensorboard_log, tb_log_name, new_tb_log};

        Runner runner{env,*act_model,n_steps,gamma,lam};

        episode_reward = Mat::Zero(n_envs,1);

        auto t_first_start = std::chrono::system_clock::now();

        int n_updates = total_timesteps / n_batch;

        for (int update = 1; update<=n_updates; ++update) {
            assert((n_batch % nminibatches) == 0);

            int batch_size = n_batch / nminibatches;
            auto t_start = std::chrono::system_clock::now();

            const MiniBatch &mb = runner.run();
            const auto& mb_all = mb.get_all();
            //all batch vectors have same num of rows
            auto num_rows = mb_all[0]->rows();
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm{num_rows};
            perm.setIdentity();

            num_timesteps += n_batch;

            std::shared_ptr<Mat> mb_loss_vals {std::make_shared<Mat>(noptepochs*nminibatches,5)};

            int update_fac = n_batch / nminibatches / noptepochs + 1;

            for (int epoch_num = 0; epoch_num<noptepochs; epoch_num++) {
                //std::random_shuffle(inds.begin(), inds.end());

                //omit last one: true rewards
                std::vector<std::shared_ptr<Mat>> mb_permutated {mb_all.size()-1};

                std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());

                //shuffle all-1 batch vectors in the same way
                for (int i = 0; i< mb_permutated.size(); ++i){
                    const auto& v = mb_all[i];
                    std::shared_ptr<Mat> tmp {std::make_shared<Mat>()};
                    *tmp = perm * *v;
                    mb_permutated[i] = tmp;
                }

                for (int start = 0; start<n_batch; start+=batch_size) {
                    int timestep = num_timesteps / update_fac + (noptepochs*n_batch + epoch_num *n_batch + start)/batch_size;

                    //slices
                    std::vector<Mat> slices {mb_permutated.size()};

                    for (int i = 0; i< mb_permutated.size(); ++i){
                        slices[i] = mb_permutated[i]->block(start,0,batch_size,1);
                    }

                    //TRAIN STEPS
                    const Mat& losses = _train_step(
                            learning_rate,
                            cliprange,
                            slices[0],
                            slices[1],
                            slices[2],
                            slices[3],
                            slices[4],
                            slices[5],
                            timestep,
                            writer);

                    int loss_index = start/batch_size + epoch_num*nminibatches;

                    mb_loss_vals->block(loss_index,0,1,5) = losses;
                }
            }
            //loss_vals = np.mean(mb_loss_vals, axis=0)

            auto loss_vals = mb_loss_vals->colwise().mean();

            auto t_now = std::chrono::system_clock::now();

            int fps = static_cast<int>(n_batch / (t_now - t_start).count());

            std::cout << fps << ",";

            for (int i = 0; i < 5; ++i){
                std::cout << loss_vals(i,0) << ",";
            }

            if(!tb_log_name.empty()){
                Eigen::Map<Mat> rewards_view(mb.true_rewards->data(), n_envs, n_steps);
                Eigen::Map<Mat> dones_view(mb.dones->data(), n_envs, n_steps);

                episode_reward = Utils::total_episode_reward_logger(episode_reward,rewards_view,dones_view,writer,num_timesteps);
            }
        }

    }

private:
    Mat _train_step(float learning_rate,
                     float cliprange,
                     const Mat& obs,
                     const Mat& returns,
                     const Mat& masks,
                     const Mat& actions,
                     const Mat& values,
                     const Mat& neglogpacs,
                     int update,
                     const TensorboardWriter& writer
    ) {
        Mat losses{5,1};

        return losses;
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

    Mat obs;
    Mat episode_reward;

};


#endif //PPO_CPP_PPO2_HPP
