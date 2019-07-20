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


void handle_segfault_signal(int sig) {

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

void playback(Env& e_norm, PPO2& algorithm){
    Mat obs{e_norm.reset()};
    float episode_reward = 0;
    while (true){
        std::cout << "obs: " << obs << std::endl;
        Mat a = algorithm.eval(obs);
        std::vector<Mat> outputs = e_norm.step(a);
        obs = std::move(outputs[0]);
        e_norm.render();
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
    signal(SIGSEGV, handle_segfault_signal);

    args::ArgumentParser parser("This is a gait viewer program.", "This goes after the options.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> load_path(parser, "path", "Serialized model to visualize", {'p',"path"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    auto seconds = time (nullptr);
    std::string run_id {"ppo_"+std::to_string(seconds)};
    std::string tb_path {"./exp/ppo_cpp/tensorboard/"+run_id+"/"};

    bool training = !load_path;
//    std::cout << "load_path: " << load_path.Get() << std::endl;
//    std::cout << "training: " << training << std::endl;

    load_and_init_robot2();
    HexapodEnv e {1};
    EnvNormalize e_norm{e,training};
    PPO2 algorithm {"./exp/ppo_cpp/resources/ppo2_graph.meta.txt",e_norm,
                    0.99,2048,0,1e-3,0.5f,.5,.95,32,10,0.2,-1,tb_path
    };

    if(training) {
        //shell-dependant timestamped directory creation
        std::string mkdir_sys_call {"mkdir -p "+tb_path};
        system(mkdir_sys_call.c_str());

        algorithm.learn(static_cast<int>(2e4));

        std::string checkpoint_path{"./exp/ppo_cpp/checkpoints/" + run_id + ".pkl"};

        algorithm.save(checkpoint_path);
        e_norm.save(checkpoint_path);


    } else {
        algorithm.load(load_path.Get());
        e_norm.load(load_path.Get());

        playback(e_norm,algorithm);
    }



    return 0;
}
