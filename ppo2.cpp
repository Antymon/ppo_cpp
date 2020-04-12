//
// Created by szymon on 10/07/19.
//

#include <execinfo.h>
#include <csignal>
#include <fstream>
#include <limits>
#include <chrono>
#include <thread>
#include <cmath>

#include "args.hxx"

#include "ppo2/ppo2.hpp"

#include "env/env.hpp"
#include "env/vec_env.hpp"
#include "env/env_mock.hpp"
#include "env/hexapod_env.hpp"
#include "env/env_normalize.hpp"
#include "env/hexapod_closed_loop_env.hpp"

void handle_signal(int sig) {

    const int stack_size = 50;

    void *array[stack_size];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, stack_size);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

void playback(Env& env, PPO2& algorithm, bool verbose, const int steps, const int framerate){

    const float frameTime = 1000/framerate;

    Mat obs{env.reset()};
    float episode_reward = 0;
    for(int i = 0; i < steps; ++i){
        //measure frame duration
        auto start = std::chrono::steady_clock::now();

        if(verbose) {
            std::cout << "obs: " << env.get_original_obs() << std::endl;
        }
        Mat a = algorithm.eval(obs);
        std::vector<Mat> outputs = env.step(a);

        obs = std::move(outputs[0]);
        float rew = env.get_original_rew()(0,0);
        env.render();
        if(verbose) {
            std::cout << "step reward: " << rew << std::endl;
        }

        episode_reward+= rew;
        if(outputs[2](0,0)>.5 || i==steps-1){
            std::cout << "episode reward: " << episode_reward << std::endl;
            episode_reward = 0;
        }

        //sleep if frame was processed too fast for visualization framerate request
        auto end = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if(duration<frameTime) {
            int x = std::round(frameTime-duration);
            std::this_thread::sleep_for(std::chrono::milliseconds(x));
        }
    }
}

void mkdir(const std::string& path){
    std::string mkdir_sys_call {"mkdir -p "+path};
    int mkdir_result {system(mkdir_sys_call.c_str())};
    if(mkdir_result == -1){
        std::cout << "Path creation failed, terminating: " << path << std::endl;
        assert(false);
    }
}

int main(int argc, char **argv)
{
    signal(SIGSEGV, handle_signal);
    signal(SIGABRT, handle_signal);

    args::ArgumentParser parser("This is a gait learner/viewer program using PPO algorithm", "--END--");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});


    args::ValueFlag<std::string> save_path(parser, "save path", "directory to save all serializations and logs", {'d',"dir"},"./exp/ppo_cpp");

    args::ValueFlag<std::string> graph_path(parser, "graph path", "path of computational graph to load", {'g',"graph","graph_path"},""); //,"./exp/ppo_cpp/resources/ppo2_graph.meta.txt");

    args::ValueFlag<std::string> load_path(parser, "checkpoint prefix", "serialized model to visualize", {'p',"path"});

    args::ValueFlag<std::string> id(parser, "unique id", "outer/global id, necessarily unique or results overrides will happen", {"id"});

    args::ValueFlag<float> steps(parser, "steps", "Total number of training steps", {'s',"steps"},2e7);

    args::ValueFlag<float> learning_rate(parser, "learning rate", "Adam optimizer's learning rate", {'l',"lr","learning_rate","learningrate"},1e-3);
    args::ValueFlag<float> entropy(parser, "entropy", "Entropy to encourage exploration", {'e',"ent","entropy"},0);
    args::ValueFlag<float> clip_range(parser, "clip range", "PPO's maximal relative change of policy likelihood", {'c',"cr","clip_range","cliprange"},0.2);

    args::ValueFlag<int> num_saves(parser, "num saves", "Number of saves. If not defined max(1 per 1M steps, 1)", {"saves","n_saves","num_saves"});
    args::ValueFlag<int> num_epochs(parser, "num epochs", "Number of epochs to train with batch of data.", {"epochs","n_epochs","num_epochs"},10);

    args::ValueFlag<int> num_batch_steps(parser, "batch steps per env", "Number of steps taken for each batch for each environment", {"batch_steps","n_steps","num_steps"},2048);

    args::ValueFlag<double>reset_noise_scale(parser,"reset noise amplitude", "Maximal amplitude of iid noise added upon reset.",{"reset_noise_scale","reset_noise","rns","rn"},0.1);
    args::Flag closed_loop(parser,"closed loop environment", "If set, closed-loop hexapod environment will be used, open-loop by default",{"closed_loop","closed-loop","cl"});

    args::Flag verbose(parser,"verbose", "output additional logs to the console",{'v',"verbose"});
    args::Flag resume(parser,"resume", "flag signalling resuming",{'r',"resume"});

    args::Flag use_bullet(parser,"use_bullet", "Replace default constraint solver with Bullet",{"bullet","use_bullet","bullet_solver"});

    args::ValueFlag<double> duration(parser, "duration", "The total duration of played animation [seconds]", {"duration","du"},5.);

    args::ValueFlag<int> threads(parser, "num threads", "Number of threads used in training", {'j',"jobs","threads","n_threads","num_threads","nt"},1);

    args::ValueFlag<int> framerate(parser, "framerate", "Framerate of visualization, no effect on training", {'f',"framerate", "fps"},60);

    //seeding needs fixing
//    args::ValueFlag<int> seed(parser, "seed", "Seed. Time-based if not specified.", {"seed"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (const args::ValidationError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    //still not deterministic - perhaps TF needs a global seed setter on the graph
//    if (seed){
//        srand(seed.Get());
//    } else {
        auto now = std::chrono::high_resolution_clock::now();
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        int seed_val = static_cast<int>(nanos % std::numeric_limits<int>::max());
        srand(seed_val);
//    }

//    std::cout << "seed: " << seed << std::endl;

    auto seconds = time (nullptr);
    std::string run_id {id?id.Get():("ppo_"+std::to_string(seconds))};
    std::string tb_path {save_path.Get()+"/tensorboard/"+run_id+"/"};

    bool training = !load_path || resume;
//    std::cout << "load_path: " << load_path.Get() << std::endl;
//    std::cout << "training: " << training << std::endl;

    load_and_init_robot2();

    std::unique_ptr<Env> wrapped_env;
    std::vector<std::shared_ptr<Env>> envs;

    bool multi_env = threads.Get()>1;

#ifdef GRAPHIC
    if (multi_env){
        std::cout << "WARNING: Visuals enabled in a multithreaded mode. Is this intentional?" << std::endl;
    }
#endif

    if(multi_env){
        for (int i =0; i<threads.Get(); ++i){
            //TODO: environment selection should be recoverable from serialization as well
            if(closed_loop){
                envs.push_back(std::make_shared<HexapodClosedLoopEnv>(reset_noise_scale.Get(),!multi_env, use_bullet));
            } else {
                envs.push_back(std::make_shared<HexapodEnv>(!multi_env, use_bullet));
            }
        }
        wrapped_env = std::make_unique<VecEnv>(envs);
    } else {
        //TODO: environment selection should be recoverable from serialization as well
        if(closed_loop){
            wrapped_env = std::make_unique<HexapodClosedLoopEnv>(reset_noise_scale.Get(),!multi_env, use_bullet);
        } else {
            wrapped_env = std::make_unique<HexapodEnv>(!multi_env, use_bullet);
        }
    }

    EnvNormalize env{std::move(wrapped_env),training};

    const std::string final_graph_path{graph_path.Get()};

    std::cout << "lr: " << learning_rate.Get() << std::endl;
    std::cout << "ent: " << entropy.Get() << std::endl;
    std::cout << "cr: " << clip_range.Get() << std::endl;

    PPO2 algorithm {final_graph_path,env,
                    .99,num_batch_steps.Get(),entropy.Get(),learning_rate.Get(),.5,.5,.95,32,num_epochs.Get(),clip_range.Get(),-1,tb_path
    };


    if(load_path){
        algorithm.load(load_path.Get());
    }

    if(training) {
        //shell-dependant timestamped directory creation

        mkdir(tb_path);

        std::string checkpoint_dir{save_path.Get()+"/checkpoints/"+run_id+"/"};

        mkdir(checkpoint_dir);

        std::string checkpoint_path{checkpoint_dir+"/" + run_id + ".pkl"};


        int int_steps {static_cast<int>(steps.Get())};

        int total_saves;

        if(num_saves){
            total_saves = num_saves.Get();
        } else {
            //max(1 per million, 1)
            total_saves = int_steps>1e6? static_cast<int>(int_steps/1e6):1;
        }

        std::cout << "steps: " << int_steps << std::endl;
        std::cout << "num_saves: " << total_saves << std::endl;

        algorithm.learn(int_steps,total_saves,checkpoint_path);

    } else {

        const int playback_steps = static_cast<int>(duration.Get()/0.015);

        playback(env,algorithm,verbose.Get(), playback_steps, framerate.Get());
    }

    global2::global_robot.reset();

    return 0;
}
