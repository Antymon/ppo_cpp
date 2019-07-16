//
// Created by szymon on 10/07/19.
//

#include "ppo2/ppo2.hpp"
#include "env/env.hpp"
#include "env/env_mock.hpp"
#include "env/hexapod_env.hpp"
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

int main(){

    signal(SIGSEGV, handle_segfault_signal);

    //shell-dependant timestamped directory creation
    auto seconds = time (nullptr);
    std::string tb_path {"./exp/ppo_cpp/tensorboard/ppo_"+std::to_string(seconds)+"/"};
    std::string mkdir_sys_call {"mkdir -p "+tb_path};
    system(mkdir_sys_call.c_str());

    load_and_init_robot2();

    HexapodEnv e {1};
    PPO2 algorithm {"./exp/ppo_cpp/resources/ppo2_graph.meta.txt",e,
                    0.99,2048,0,1e-3,0.5f,.5,.95,32,10,0.2,-1,tb_path
    };

    algorithm.learn(static_cast<int>(2e7));

    return 0;
}
