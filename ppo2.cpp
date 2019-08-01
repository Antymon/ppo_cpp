//
// Created by szymon on 10/07/19.
//

#include "ppo2/ppo2.hpp"
#include "env/env.hpp"
#include "env/env_mock.hpp"
#include "env/hexapod_env.hpp"
#include "env/env_normalize.hpp"
#include "args.hxx"
#include <execinfo.h>
#include <signal.h>
#include "json.hpp"
#include <fstream>


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

void playback(Env& env, PPO2& algorithm){
    Mat obs{env.reset()};
    float episode_reward = 0;
    while (true){
        std::cout << "obs: " << obs << std::endl;
        Mat a = algorithm.eval(obs);
        std::vector<Mat> outputs = env.step(a);
        obs = std::move(outputs[0]);
        env.render();
        std::cout << "step reward: " << outputs[1] << std::endl;
        episode_reward+= outputs[1](0,0);
        if(outputs[2](0,0)>.5){
            std::cout << "episode reward: " << episode_reward << std::endl;
            episode_reward = 0;
        }
    }
}



int main(int argc, char **argv)
{
    signal(SIGSEGV, handle_signal);
    signal(SIGABRT, handle_signal);

    args::ArgumentParser parser("This is a gait viewer program.", "This goes after the options.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});


    args::ValueFlag<std::string> save_path(parser, "save path", "directory to save all serializations and logs", {'d',"dir"},"./exp/ppo_cpp");

    args::ValueFlag<std::string> graph_path(parser, "graph path", "path of computational graph to load", {'g',"graph","graph_path"},"./exp/ppo_cpp/resources/ppo2_graph.meta.txt");

    args::ValueFlag<std::string> load_path(parser, "checkpoint prefix", "Serialized model to visualize", {'p',"path"});

    args::ValueFlag<float> steps(parser, "steps", "Total number of training steps", {'s',"steps"},2e7);

    args::ValueFlag<float> learning_rate(parser, "learning rate", "Adam optimizer's learning rate", {'l',"lr","learning_rate","learningrate"},1e-3);
    args::ValueFlag<float> entropy(parser, "entropy", "Entropy to encourage exploration", {'e',"ent","entropy"},0);
    args::ValueFlag<float> clip_range(parser, "clip range", "PPO's maximal relative change of policy likelihood", {'c',"cr","clip_range","cliprange"},0.2);

    args::ValueFlag<int> num_saves(parser, "num saves", "Number of saves. If not defined max(1 per 1M steps, 1)", {"saves","n_saves","num_saves"});
    args::ValueFlag<int> num_epochs(parser, "num epochs", "Number of epochs to train with batch of data.", {"epochs","n_epochs","num_epochs"},10);

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

    auto seconds = time (nullptr);
    std::string run_id {"ppo_"+std::to_string(seconds)};
    std::string tb_path {save_path.Get()+"/tensorboard/"+run_id+"/"};

    bool training = !load_path;
//    std::cout << "load_path: " << load_path.Get() << std::endl;
//    std::cout << "training: " << training << std::endl;

    load_and_init_robot2();
    HexapodEnv inner_e {1};
    EnvNormalize env{inner_e,training};

    const std::string final_graph_path{graph_path.Get()};

    std::cout << "lr: " << learning_rate.Get() << std::endl;
    std::cout << "ent: " << entropy.Get() << std::endl;
    std::cout << "cr: " << clip_range.Get() << std::endl;

    PPO2 algorithm {final_graph_path,env,
                    .99,2048,entropy.Get(),learning_rate.Get(),.5,.5,.95,32,num_epochs.Get(),clip_range.Get(),-1,tb_path
    };

    if(training) {
        //shell-dependant timestamped directory creation
        std::string mkdir_sys_call {"mkdir -p "+tb_path};
        system(mkdir_sys_call.c_str());

        std::string checkpoint_dir{save_path.Get()+"/checkpoints"};
        mkdir_sys_call = "mkdir -p "+ checkpoint_dir;
        system(mkdir_sys_call.c_str());

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
        algorithm.load(load_path.Get());

        playback(env,algorithm);
    }



    return 0;
}
