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

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

const std::string BasePolicy::obs_ph{"input/Ob:0"};
const std::string ActorCriticPolicy::action{"output/_action:0"};
const std::string ActorCriticPolicy::neglogp{"output/_neglogp"};
const std::string ActorCriticPolicy::value_flat{"output/_value_flat"};

struct MiniBatch {
    Mat obs;
    Mat returns;
    Mat dones;
    Mat actions;
    Mat values;
    Mat neglogpacs;
};

class Runner {
public:
    Runner(Env& env, MlpPolicy& model, int n_steps, float gamma, float lam)
    : env{env}
    , model{model}
    , n_steps{n_steps}
    , gamma {gamma}
    , lam{lam}{}

    MiniBatch run() {
        MiniBatch mb;

        return mb;
    }
private:
    Env& env;
    MlpPolicy& model;
    int n_steps;
    float gamma;
    float lam;
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

        Runner runner{env,*act_model.get(),n_steps,gamma,lam};

        auto t_first_start = std::chrono::system_clock::now();

        int n_updates = total_timesteps / n_batch;

        for (int update = 1; update<=n_updates; ++update) {
            assert((n_batch % nminibatches) == 0);

            int batch_size = n_batch / nminibatches;
            auto t_start = std::chrono::system_clock::now();

            const MiniBatch &mb = runner.run();

            num_timesteps += n_batch;

            //mb_loss_vals{};

            int update_fac = n_batch / nminibatches / noptepochs + 1;

            std::vector<int> inds(n_batch);
            for (int i=0; i<n_batch; ++i) {
                inds[i]=i;
            }
            for (int epoch_num = 0; epoch_num<noptepochs; epoch_num++) {
                std::random_shuffle(inds.begin(), inds.end());
                for (int start = 0; start<n_batch; start+=batch_size) {
                    int timestep = num_timesteps / update_fac + (noptepochs*n_batch + epoch_num *n_batch + start)/batch_size;

                    int end = start + batch_size;

                    std::vector<int> mbinds(inds.begin() + start, inds.begin() + end);

                    //slices

//                    std::vector<int> obs(mbinds.size());
//                    for (size_t i = 0; i < mbinds.size(); ++i )
//                        obs[i] = A[mbinds[i]];

                    //TRAIN STEPS
                }
            }
            //loss_vals = np.mean(mb_loss_vals, axis=0)
            auto t_now = std::chrono::system_clock::now();

            int fps = static_cast<int>(n_batch / (t_now - t_start).count());

            //writer
        }

    }

private:
    void _train_step(float learning_rate,
                     float cliprange,
                     Mat obs,
                     Mat returns,
                     Mat masks,
                     Mat actions,
                     Mat values,
                     Mat neglogpacs,
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
