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
#include <string>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include "base_class.hpp"
#include "../env/env.hpp"
#include "session_creator.hpp"
#include "policies.hpp"
#include "utils.hpp"
#include "tensorboard.hpp"
#include "runner.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

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
    , model_filename{model_filename}
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
    graph_def{} {

        //TODO this allows PPO be uninitialized
        //do loading through constructor
        if(model_filename.empty()) {
            std::cout << "PPO unitialized " << std::endl;
        } else {
            reset();
        }
    }

    void reset(){
        n_batch = n_envs * n_steps;

        std::cout << "ppo2 " << std::endl;

        SessionCreator sc{};

        _session = std::move(sc.load_graph(model_filename,graph_def));

        if (_session == nullptr || !_session) {
            std::cout<<"session cannot be nullptr"<<std::endl;
            assert(false);
        }

        act_model = std::make_unique<MlpPolicy>(_session);
    }

    void save(std::string save_path, int save_id = -1) {

        //add postfix when number of saves
        if(save_id >=0){
            save_path += "." + std::to_string(save_id);
        }

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
        if (!status.ok()) {
            std::cout << "Error saving checkpoint to " << save_path << ": " << status.ToString() << std::endl;
            assert(false);
        }
        else {
            std::cout << "Success save weights !! " << "\n";

            nlohmann::json json{};

            env.serialize(json);

            json["gamma"] = gamma;
            json["n_steps"] = n_steps;
            json["vf_coef"] = vf_coef;
            json["ent_coef"] = ent_coef;
            json["max_grad_norm"] = max_grad_norm;
            json["learning_rate"] = learning_rate;
            json["lam"] = lam;
            json["nminibatches"] = nminibatches;
            json["noptepochs"] = noptepochs;
            json["cliprange"] = cliprange;
            json["cliprange_vf"] = cliprange_vf;
            json["observation_space"] = observation_space;
            json["action_space"] = action_space;
            json["n_envs"] = n_envs;
            json["model_filename"] = model_filename;
            //json["policy_kwargs"] = policy_kwargs; //embedded in graph

            std::ofstream myfile (save_path + ".json");
            if (myfile.is_open())
            {
                myfile << json.dump();
                myfile.close();
            }
            else {
                std::cout << "Unable to open file for saving";
                assert(false);
            }

        }
    }

    //warning: does not recover
    void load(std::string save_path){

        std::ifstream in (save_path+".json");
        if (in.is_open())
        {
            std::stringstream sstr;
            sstr << in.rdbuf();
            nlohmann::json json = nlohmann::json::parse(sstr.str());
            env.deserialize(json);

            gamma = json["gamma"].get<float>();
            n_steps = json["n_steps"].get<int>();
            vf_coef = json["vf_coef"].get<float>();
            ent_coef = json["ent_coef"].get<float>();
            max_grad_norm = json["max_grad_norm"].get<float>();
            learning_rate = json["learning_rate"].get<float>();
            lam = json["lam"].get<float>();
            nminibatches = json["nminibatches"].get<int>();
            noptepochs = json["noptepochs"].get<int>();
            cliprange = json["cliprange"].get<float>();
            cliprange_vf = json["cliprange_vf"].get<float>();
            observation_space = json["observation_space"].get<std::string>();
            action_space = json["action_space"].get<std::string>();
            n_envs = json["n_envs"].get<int>();
            if (model_filename.empty()) {
                model_filename = json["model_filename"].get<std::string>();
            } else {
                std::cout << "filename passed through CLI overrides deserialized one" << std::endl;
            }

            reset();

            in.close();
        }
        else{
            std::cout << "Unable to open file for loading"<< std::endl;
            assert(false);
        }


        tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
        checkpointPathTensor.scalar<std::string>()() = save_path;
        std::vector<std::pair<std::string,tensorflow::Tensor>> feed_dict = {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
        auto status = _session->Run(feed_dict, {}, {graph_def.saver_def().restore_op_name()}, nullptr);

        if (!status.ok()) {
            std::cout << "Error loading checkpoint from " << save_path << ": " << status.ToString() << std::endl;
            assert(false);
        }
        else {
            std::cout << "Success load weights !! "<< std::endl;


        }
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

    void learn(int total_timesteps, int num_saves=0, const std::string& save_path = "", const std::string& tb_log_name = "PPO2") {

        if(!_session){
            std::cout << "Session unitialized, learning aborted.";
            assert(false);
        }

        bool new_tb_log = _init_num_timesteps();

        TensorboardWriter writer {tensorboard_log, tb_log_name, new_tb_log};

        Runner runner{env,*act_model,n_steps,gamma,lam};

        episode_reward = Mat::Zero(n_envs,1);

        auto t_first_start = std::chrono::system_clock::now();

        int n_updates = total_timesteps / n_batch;

        int save_interval = -1;
        if(num_saves > 0) {
            save_interval = static_cast<int>(std::ceil(
                    static_cast<float>(n_updates) / static_cast<float>(num_saves)));
        }

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

            if(save_interval>0 && num_saves > 0 && update%save_interval == 0){
                assert(!save_path.empty());
                save(save_path,update/save_interval-1);
            }
        }


        if(num_saves>0) {
            assert(!save_path.empty());
            //one more save if interval didn't evenly divide num updates
            if (save_interval > 0 && ((n_updates % save_interval) != 0)) {
                save(save_path, n_updates / save_interval);
            } else if(save_interval==0){
                save(save_path);
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
                     TensorboardWriter& writer
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

        std::vector<tensorflow::Tensor> tensor_outputs;

        tensorflow::Status s = _session->Run(td_map,output_placeholders, {_train}, &tensor_outputs);

        if (!s.ok()) {
            std::cout << "train error" << std::endl;
            std::cout << s.ToString() << std::endl;
            assert(false);
        }


        //need some metadata stripping like in python implementation to make this feasible size-wise
//        auto encoded_summary {tensor_outputs[0].scalar<std::string>()()};
//        int update_fac = n_batch /nminibatches/noptepochs + 1;
//        writer.write_summary(update_fac*update,std::move(encoded_summary));

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

    std::string model_filename;

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
