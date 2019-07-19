#include <utility>

//
// Created by szymon on 10/07/19.
//
// Graph Reading cpp counterpart of modified PPO2 implementation from Stable-Baselines
// It executes training but the graph needs to be read from outside
//

#ifndef PPO_CPP_PPO2_HPP
#define PPO_CPP_PPO2_HPP

#include <utility>
#include <memory>
#include "Eigen/Dense"
#include <string>
#include "base_class.hpp"
#include "../env/env.hpp"
#include <iostream>
#include "session_creator.hpp"
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

        //srarts with num envs!
        mb.obs = std::make_shared<Mat>(num_envs,n_steps*env.get_observation_space_size());
        mb.returns = std::make_shared<Mat>(n_steps, num_envs);
        mb.dones = std::make_shared<Mat>(n_steps, num_envs);

        //srarts with num envs!
        mb.actions = std::make_shared<Mat>(num_envs,n_steps*env.get_action_space_size());
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

    void set_returns(MiniBatch& mb){

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
         float cliprange_vf = -1.,
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
    , cliprange_vf{cliprange_vf}
    , tensorboard_log{std::move(tensorboard_log)}
    , episode_reward{Mat()}
    /*, input_placeholders{
        train_obs_ph,
        action_ph,
        advs_ph,
        rewards_ph,
        learning_rate_ph,
        clip_range_ph,
        old_neglog_pac_ph,
        old_vpred_ph
    }*/,output_placeholders{
        summary,
        pg_loss,
        vf_loss,
        entropy,
        approxkl,
        clipfrac
    },
    graph_def{}
    {
        n_batch = n_envs * n_steps;

        std::cout << "ppo2 " << std::endl;

        SessionCreator sc{};
        _session = std::move(sc.load_graph(std::move(model_filename),graph_def));

        if (_session == nullptr || !_session) {
            std::cout<<"session cannot be nullptr"<<std::endl;
            assert(false);
        }

        act_model = std::make_unique<MlpPolicy>(_session);
    }

    void save(std::string save_path) {
        tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
        checkpointPathTensor.scalar<std::string>()() = save_path;

        std::cout << graph_def.saver_def().filename_tensor_name() << "\n";
        std::cout << graph_def.saver_def().save_tensor_name() << "\n";
        std::cout << save_path << "\n";

        auto status = _session->Run(
                {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor},},
                {},
                {graph_def.saver_def().save_tensor_name()},
                nullptr);
        if (!status.ok())
            std::cout << "Error saving checkpoint to " << save_path << ": " << status.ToString() << std::endl;
        else
            std::cout << "Success save weights !! " << "\n";
    }

    void load(std::string save_path){
        tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
        checkpointPathTensor.scalar<std::string>()() = save_path;
        std::vector<std::pair<std::string,tensorflow::Tensor>> feed_dict = {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
        auto status = _session->Run(feed_dict, {}, {graph_def.saver_def().restore_op_name()}, nullptr);

        if (!status.ok())
            std::cout << "Error loading checkpoint from " << save_path << ": " << status.ToString() << std::endl;
        else
            std::cout << "Success load weights !! " << "\n";
    }

    Mat eval(const Mat &obs) const {

        auto tensor_obs = tensorflow::Tensor();
        Utils::convert_mat(obs,tensor_obs);

        tensorflow::Tensor tensor_actions =  act_model->get_action(tensor_obs);

        Mat actions;
        Utils::convert_tensor(tensor_actions,actions,env.get_action_space());
        assert(actions.rows()==1 && actions.cols()==env.get_action_space_size());

        return actions;
    }

    void learn(int total_timesteps, const std::string& tb_log_name = "PPO2") {

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
            const auto& mb_all = mb.get_train_input();
            //all batch vectors have same num of rows
            auto num_rows = mb_all[0]->rows();
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm{num_rows};
            perm.setIdentity();

            num_timesteps += n_batch;

            std::shared_ptr<Mat> mb_loss_vals {std::make_shared<Mat>(noptepochs*nminibatches,5)};

            int update_fac = n_batch / nminibatches / noptepochs + 1;

            for (int epoch_num = 0; epoch_num<noptepochs; epoch_num++) {
                //std::random_shuffle(inds.begin(), inds.end());

                std::vector<std::shared_ptr<Mat>> mb_permutated {mb_all.size()};

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
                        //get consecutive rows from range, with all columns
                        slices[i] = mb_permutated[i]->block(start,0,batch_size,mb_all[i]->cols());
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

//                    std::cout << "#6" << std::endl;
//                    std::cout << losses << std::endl;
//                    std::cout << loss_index << std::endl;

                    mb_loss_vals->block(loss_index,0,1,5) = losses;
                }

                //std::cout << "epoch " << epoch_num << std::endl;
            }
//            std::cout << "#7" << std::endl;

            auto loss_vals = mb_loss_vals->colwise().mean();

            auto t_now = std::chrono::system_clock::now();

            auto duration = (std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start)).count();

            int fps = static_cast<int>(n_batch*1000 / duration);

            std::cout << fps << ",";

            for (int i = 0; i < 5; ++i){
                std::cout << loss_vals(0,i) << ",";
            }

            std::cout << std::endl;

            if(!tb_log_name.empty()){

                //assert((*(mb.unnormalized_rewards)-*(mb.true_rewards)).cwiseAbs().sum() < 1e-3);

                Eigen::Map<Mat> rewards_view(mb.unnormalized_rewards->data(), n_envs, n_steps);
                Eigen::Map<Mat> dones_view(mb.dones->data(), n_envs, n_steps);

                episode_reward = Utils::total_episode_reward_logger(episode_reward,rewards_view,dones_view,writer,num_timesteps-n_batch);
            }
        }

    }

private:
    Mat _train_step(float learning_rate_now,
                     float cliprange_now,
                     const Mat& obs,
                     const Mat& returns,
                     const Mat& masks,
                     const Mat& actions,
                     const Mat& values,
                     const Mat& neglogpacs,
                     int update,
                     const TensorboardWriter& writer
    ) {

        assert(obs.cols() == env.get_observation_space_size());
        assert(actions.cols() == env.get_action_space_size());

        Mat learning_rate_wrapper{1,1};
        learning_rate_wrapper(0,0) = learning_rate_now;

        Mat clip_range_wrapper{1,1};
        clip_range_wrapper(0,0) = cliprange_now;

        Mat advs {returns-values};
        assert(advs.rows()>1 && advs.cols()==1);
        float advs_mean = advs.mean();
        Mat advs_sub_mean {advs - advs_mean*Mat::Ones(advs.rows(),advs.cols())};
        float advs_var = (advs_sub_mean.cwiseProduct(advs_sub_mean)).sum()/advs.rows();
        advs = advs_sub_mean / (std::sqrt(advs_var) + 1e-8);

        tensorflow::Tensor obs_tensor{};
        tensorflow::Tensor returns_tensor{};
        tensorflow::Tensor actions_tensor{};
        tensorflow::Tensor values_tensor{};
        tensorflow::Tensor neglogpacs_tensor{};
        tensorflow::Tensor advs_tensor{};
        tensorflow::Tensor learning_rate_tensor{};
        tensorflow::Tensor clip_range_tensor{};

        Utils::convert_mat(obs,obs_tensor);
        Utils::convert_mat(actions,actions_tensor);

        Utils::convert_vec(returns,returns_tensor);
        Utils::convert_vec(values,values_tensor);
        Utils::convert_vec(neglogpacs,neglogpacs_tensor);
        Utils::convert_vec(advs,advs_tensor);

        Utils::scalar(learning_rate_now,learning_rate_tensor);
        Utils::scalar(cliprange_now, clip_range_tensor);

        //std::cout << advs_tensor.dims() << std::endl;

        std::vector<std::pair<std::string,tensorflow::Tensor>> td_map =
                {
                        {train_obs_ph,obs_tensor},
                        {action_ph,actions_tensor},
                        {advs_ph,advs_tensor},
                        {rewards_ph,returns_tensor},
                        {learning_rate_ph,learning_rate_tensor},
                        {clip_range_ph,clip_range_tensor},
                        {old_neglog_pac_ph,neglogpacs_tensor},
                        {old_vpred_ph,values_tensor}
                };

        if (cliprange_vf >= 0.f){
            tensorflow::Tensor clip_range_vf_tensor{};
            Utils::scalar(cliprange_vf,clip_range_vf_tensor);
            td_map.emplace_back(clip_range_vf_ph,clip_range_vf_tensor);
        }

        //unitl tensor board fully implemented
        //int update_fac = n_batch /nnminibatches / noptepochs + 1;

        std::vector<tensorflow::Tensor> tensor_outputs;

        tensorflow::Status s = _session->Run(td_map,output_placeholders, {_train}, &tensor_outputs);

        if (!s.ok()) {
            std::cout << "train error" << std::endl;
            std::cout << s.ToString() << std::endl;
            assert(false);
        }

        //summary proto string that needs to be decoded and fed into tb
        //std::cout << tensor_outputs[0].scalar<std::string>()() << std::endl;

        Mat losses{1,5};

        for(int i = 1; i<tensor_outputs.size(); ++i){
            losses(0,i-1)=tensor_outputs[i].scalar<float>()();
        }

        return losses;
    }

    static const std::string train_obs_ph;
    static const std::string action_ph;
    static const std::string advs_ph;
    static const std::string rewards_ph;
    static const std::string learning_rate_ph;
    static const std::string clip_range_ph;
    static const std::string old_neglog_pac_ph;
    static const std::string old_vpred_ph;
    //opt input
    static const std::string clip_range_vf_ph;


    static const std::string summary;
    static const std::string pg_loss;
    static const std::string vf_loss;
    static const std::string entropy;
    static const std::string approxkl;
    static const std::string clipfrac;
    static const std::string _train;

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
    float cliprange_vf;
    std::string tensorboard_log;
    int n_batch;

    std::shared_ptr<tensorflow::Session> _session;
    std::unique_ptr<MlpPolicy> act_model;

    Mat episode_reward;

    const std::vector<std::string> output_placeholders;
    tensorflow::MetaGraphDef graph_def;

};

const std::string BasePolicy::obs_ph{"input/Ob:0"};

const std::string ActorCriticPolicy::action{"output/_action:0"};
const std::string ActorCriticPolicy::neglogp{"output/_neglogp:0"};
const std::string ActorCriticPolicy::value_flat{"output/_value_flat:0"};

const std::string PPO2::train_obs_ph{"train_model/input/Ob:0"};
const std::string PPO2::action_ph{"loss/action_ph:0"};
const std::string PPO2::advs_ph{"loss/advs_ph:0"};
const std::string PPO2::rewards_ph{"loss/rewards_ph:0"};
const std::string PPO2::learning_rate_ph{"loss/learning_rate_ph:0"};
const std::string PPO2::clip_range_ph{"loss/clip_range_ph:0"};
const std::string PPO2::old_neglog_pac_ph{"loss/old_neglog_pac_ph:0"};
const std::string PPO2::old_vpred_ph{"loss/old_vpred_ph:0"};
const std::string PPO2::clip_range_vf_ph{"loss/clip_range_vf_ph:0"};

const std::string PPO2::summary{"ppo2/summary/ppo2/summary:0"};
const std::string PPO2::pg_loss{"loss/pg_loss:0"};
const std::string PPO2::vf_loss{"loss/vf_loss:0"};
const std::string PPO2::entropy{"loss/ppo2/entropy:0"};
const std::string PPO2::approxkl{"loss/approxkl:0"};
const std::string PPO2::clipfrac{"loss/clipfrac:0"};
const std::string PPO2::_train{"ppo2/_train"};


#endif //PPO_CPP_PPO2_HPP
