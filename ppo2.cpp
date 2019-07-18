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

int main(int argc, char **argv)
{
    args::ArgumentParser parser("This is a gait viewer program.", "This goes after the options.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::ValueFlag<std::string> load_path(parser, "path", "Serialized model to visualize", {'p',"path"});

    signal(SIGSEGV, handle_segfault_signal);

    //shell-dependant timestamped directory creation
    auto seconds = time (nullptr);
    std::string run_id {"ppo_"+std::to_string(seconds)};
    std::string tb_path {"./exp/ppo_cpp/tensorboard/"+run_id+"/"};
    std::string mkdir_sys_call {"mkdir -p "+tb_path};
    system(mkdir_sys_call.c_str());

    load_and_init_robot2();

    HexapodEnv e {1};
    EnvNormalize e_norm{e};
    PPO2 algorithm {"./exp/ppo_cpp/resources/ppo2_graph.meta.txt",e_norm,
                    0.99,2048,0,1e-3,0.5f,.5,.95,32,10,0.2,-1,tb_path
    };

    if(!load_path) {
        algorithm.learn(static_cast<int>(2e7));
        algorithm.save("./exp/ppo_cpp/checkpoints/" + run_id + ".pkl");
    } else {
        algorithm.load(load_path.Get());

        Mat obs{e_norm.reset()};

        float episode_reward = 0;

        while (true){
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



    return 0;
}
