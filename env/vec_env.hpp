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

    //to allow vector of bool to be modified on separate slots in parallel without sync
    struct BoolWrapper {
        BoolWrapper():value{false}{}
        explicit BoolWrapper(bool value):value{value}{}
        bool value;
    };

    static void writeln(const std::string& msg, double delay = 0){
        if(delay > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay * 1000)));
        }
        std::cout << msg+"\n";
    }

    explicit VecEnv(const std::vector<std::shared_ptr<Env>>& envs)
        : Env()
        , envs{envs}
        , threads{}
        , slots_condition_vars{std::vector<std::condition_variable>(get_num_envs())}
        , slots_mutexes{std::vector<std::mutex>(get_num_envs())}
        , counter_mutex{}
        , counter{0}
        , all_done{}
        , slots_ready{std::vector<BoolWrapper>(get_num_envs())}
        , cached_actions {Mat::Zero(get_num_envs(),get_observation_space_size())}
        , observations {Mat::Zero(get_num_envs(),get_observation_space_size())}
        , rewards { Mat::Zero(get_num_envs(), 1)}
        , dones { Mat::Zero(get_num_envs(), 1)}
        , original_rewards { Mat::Zero(get_num_envs(), 1)}
    {
        assert(!envs.empty());

        for (unsigned long i = 0; i<envs.size(); ++i) {
            threads.emplace_back(&VecEnv::start_env_thread, this, i);
        }
    }

    //to easily avoid problems with vector of threads remove copy ops
    //move ctor and assign are not generated in such case
    VecEnv(const VecEnv&) = delete;
    VecEnv & operator=(const VecEnv&) = delete;

    virtual ~VecEnv(){
        terminate = true;
        for (unsigned long i = 0; i<envs.size(); ++i) {
            slots_condition_vars[i].notify_one();
            threads[i].join();
        }
    }

    std::string get_action_space() override {
        return envs[0]->get_action_space();
    }

    std::string get_observation_space() override {
        return envs[0]->get_observation_space();
    }

    int get_action_space_size() override{
        return envs[0]->get_action_space_size();
    }

    int get_observation_space_size() override{
        return envs[0]->get_action_space_size();
    }

    //sequential due to infrequent use
    Mat reset() override {
        Mat result = Mat::Zero(get_num_envs(),get_observation_space_size());

        for (unsigned long i = 0; i<envs.size(); ++i) {
            result.row(i) = envs[i]->reset();
        }

        return std::move(result);
    }

    std::vector<Mat> step(const Mat &actions) override {

        {
            //writeln("main requests slots locks");
            std::vector<std::unique_lock<std::mutex>> locks(get_num_envs());
            for (int i = 0; i < get_num_envs(); ++i) {
                locks[i] = std::unique_lock<std::mutex>(slots_mutexes[i]);
            }
            //writeln("main got slots locks");

            this->cached_actions=actions;

            for (int i = 0; i < get_num_envs(); ++i) {
                slots_ready[i].value = true;
            }
            std::unique_lock<std::mutex> l(counter_mutex);
            counter=0;
            l.unlock();

            for (int i = 0; i < get_num_envs(); ++i) {
                locks[i].unlock();
            }
            //writeln("main released slots locks");
        }

        //wake up all threads as actions are slots_ready to process
        //writeln("main is going to wake all");

        for (int i = 0; i<get_num_envs(); ++i){
            slots_condition_vars[i].notify_one();
        }

        //writeln("main woke all and waiting for counter mutex");

        std::unique_lock<std::mutex> l2(counter_mutex);
        while(counter < get_num_envs()) {
            all_done.wait(l2);
        }
        l2.unlock();

        //writeln("main done");
        //writeln("obs");
        //std::cout << observations << std::endl;
        //writeln("rewards");
        //std::cout << rewards << std::endl;
        return {observations,rewards,dones};
    }

    Mat get_original_obs() override {
        writeln("VecEnv::render() not implemented");
        assert(false);
        return Mat::Zero(get_num_envs(),get_observation_space_size());
    }

    Mat get_original_rew() override {
        return original_rewards;
    }

    int get_num_envs() override {
        return static_cast<int>(envs.size());
    }

    void serialize(nlohmann::json& json) override {

    }

    void deserialize(nlohmann::json& json) override {

    }

    void render() override {
        //writeln("VecEnv::render() not implemented");
        assert(false);
    }

    float get_time() override {
        //writeln("VecEnv::get_time() not implemented");
        assert(false);
        return -1;
    }

private:
    const std::vector<std::shared_ptr<Env>>& envs;
    std::vector<std::thread> threads;
    std::vector<std::condition_variable> slots_condition_vars;
    std::vector<std::mutex> slots_mutexes;
    bool terminate = false;
    std::mutex counter_mutex;
    int counter;
    std::condition_variable all_done;
    std::vector<BoolWrapper> slots_ready;
    Mat cached_actions;
    Mat observations;
    Mat rewards;
    Mat dones;
    Mat original_rewards;

    void start_env_thread(int id){
        while (!terminate){
            //wait until new action is available
            {
                //writeln(std::to_string(id) + " requests lock");
                std::unique_lock<std::mutex> l(slots_mutexes[id]);
                //writeln(std::to_string(id) + " got lock, perhaps waiting");
                slots_condition_vars[id].wait(l, [this,id]{ return slots_ready[id].value || terminate; });
                //process action by doing a single step

                if(terminate){
                    //writeln(std::to_string(id) + " terminating");
                    return;
                }

                //consume
                slots_ready[id].value = false;

                //writeln(std::to_string(id)+" doing work",.25);

                auto res = envs[id]->step(cached_actions.row(id));
                observations.row(id)=res[0];
                rewards.row(id)=res[1];
                dones.row(id)=res[2];

                original_rewards.row(id)=envs[id]->get_original_rew();

                l.unlock();

                //writeln(std::to_string(id)+" finished.");
            }

            bool notify_main;

            {
                std::unique_lock<std::mutex> l2(counter_mutex);
                if (counter < get_num_envs()) {
                    counter++;
                    notify_main = counter == get_num_envs();
                    l2.unlock();
                } else {
                    //writeln("Invariant counter<get_num_envs() violated!");
                    l2.unlock();
                    assert(false);
                }
            }

            if(notify_main){
                all_done.notify_one();
            }
        }
    }

};

#endif //PPO_CPP_VEC_ENV_HPP
