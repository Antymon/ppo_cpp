#ifndef PPO_CPP_VEC_ENV_HPP
#define PPO_CPP_VEC_ENV_HPP

#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>

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
        return 18;
    }

    int get_observation_space_size() override{
        return 18;
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

        std::unique_lock<std::mutex> l(counter_mutex);
        counter=0;
        l.unlock();

        //wake up all threads as actions are ready to process
        writeln("main wakes all");

        for (int i = 0; i<get_num_envs(); ++i){
            condition_vars[i].notify_one();
        }

        writeln("main woke all and waiting for counter mutex");

        std::unique_lock<std::mutex> l2(counter_mutex);
        if(counter < get_num_envs()) {
            all_done.wait(l2);
        }
        l2.unlock();


//        //block return unitl all threads did a step
//        for (int i = 0; i<get_num_envs(); ++i){
//            writeln("main requests lock "+std::to_string(i));
//            std::unique_lock<std::mutex> l(mutexes[i]);
//            writeln("main got lock "+std::to_string(i)+", waiting");
//            condition_vars[i].wait(l);
//            l.unlock();
//        }

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

    static void writeln(const std::string& msg, double delay = 0){
        if(delay > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay * 1000)));
        }
        std::cout << msg << std::endl;
    }

private:
    std::vector<std::shared_ptr<Env>>& envs;
    long total_step;
    std::vector<std::thread> threads;
    std::vector<std::condition_variable> condition_vars;
    std::vector<std::mutex> mutexes;
    bool terminate = false;
    std::mutex counter_mutex;
    int counter;
    std::condition_variable all_done;

    void start_env_thread(int id){
        while (!terminate){
            //wait until new action is available
            {
                writeln(id + " request lock");
                std::unique_lock<std::mutex> l(mutexes[id]);
                writeln(id + " got lock, waiting");
                condition_vars[id].wait(l);
                //process action by doing a single step

                std::cout << "obs[id] = step(a[id]) " << id << std::endl;

                l.unlock();

                writeln(id+" woken & finished.");
            }

            bool notify_main = false;

            {
                std::unique_lock<std::mutex> l2(counter_mutex);
                if (counter < get_num_envs()) {
                    writeln("Invariant counter<get_num_envs() violated!");
                    l2.unlock();
                    assert(false);
                } else {
                    counter++;
                    notify_main = counter == get_num_envs();
                    l2.unlock();
                }
            }

            if(notify_main){
                all_done.notify_one();
            }
        }
    }

};

#endif //PPO_CPP_VEC_ENV_HPP
