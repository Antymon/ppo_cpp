PPO_CPP
==========

What is it?
--------
PPO_CPP is a C++ version of a Proximal Policy Optimization algorithm @Schulman2017 with some additions.
It was partially ported from Stable Baselines @Hill2018 Deep Reinforcement Learning suite
with elements of the OpenAI Gym framework @Brockman2016, in a form of a Tensorflow @Abadi2016 graph executor. It additionally
features an example environment based on DART simulation engine @Lee2018 with a hexapod robot @Cully2015 tasked to walk as far as
possible along X axis ([example recording](https://drive.google.com/open?id=1ds_VrjTDdhqWkh40eF1vscetfUyJUlVm)).

Why?
--------
**Performance**. The interesting thing is that PPO_CPP executes 2~3 times faster than corresponding Python implementation when running on the
same environment with same number of threads. Originally, however, PPO_CPP was setup for the sake of DRL/Neuroevolution comparison in an unpublished yet project.

Is it optimized?
--------
Not at all. Particularly in the multithreaded case there might be some easy wins to further boost performance, e.g. by going away from
ported ideas and leveraging thread safety of Tensorflow's Session.Run() or promoting immutability by copying network weights.

How was it tested so far?
--------
Single-threaded version with both hexapod envs was run in many instances on a High Performance Computing cluster for a grand total of 50 000 to 100 000 CPU hours yielding believable results.

How can I use it for my work?
--------
You should be able to easily check the examples below, however if you want to use it in different settings you will probably need 3 things:
-   Make your environment inherit from Env abstract class under `env\env.hpp`
-   Modify or replace main `ppo2.cpp` which creates instance of an environment and passes it to PPO
-   Create own computational graph and potentially make some small modifications to the core algorithm if using more involved
    policies (currently implementation supports only MLP policies). Graph generation is mentioned below.

Where is the multi-threaded version?
--------
At the moment multi-threaded version lives on the *parallel* branch, because of the hexapod environment which proved to be annoyingly leaky in this setup.

State of the project
--------
This is just a proof-of-concept which could benefit from number of improvements. Let me know if this project is useful for you!

Recommended dev setup and dependencies
--------
Except for `ppo2.cpp` and potentially the hexapod environments, core PPO code should be easily portable.
Two main dependencies are Eigen @Guennebaud2010 and Tensorflow @Abadi2016. If hexapod environment considered then also DART @Lee2018.
Containers additionally use Sferes2 @Mouret2010 framework and Python WAF build system @Nagy2010 which
are outcome of history of the project and could be ditched. Most solid way to examine container dependencies
is to read project's `singularity/singularity.def` file and its parent.
To run the examples a Linux system supporting Singularity container system is needed, preferably:

-   Ubuntu 18.04 LTS operating system,

-   [Singularity](https://sylabs.io/guides/3.3/user-guide/quick_start.html#quick-installation-steps) @Kurtzer2016 containerization environment,

Examples
==========
Training gaits
--------------

To start the PPO training you need to [install
Singularity](https://sylabs.io/guides/3.3/user-guide/quick_start.html#quick-installation-steps)
@Kurtzer2016 version 3.3 or later (at the moment available only for
Linux systems). For your convenience, a well-performing PPO setup was
committed in the PPO repository. Paying attention to the very long
argument list to the SIMG file, type in bash:

    git clone https://gitlab.doc.ic.ac.uk/sb5817/ppo_cpp.git

    cd ppo_cpp/singularity

    ./build_final_image.sh

    # very long argument list
    ./final*.simg 
        0 
        ppo_cpp_[4_5]_lr_0.0004_cr_0.1610_ent_0.0007
    ../resources/ppo_cl/graphs/ppo_cpp_\[4_5\]_lr_0.0004_cr_0.1610_ent_0.0007.meta.txt 
        --steps 75000000 
        --num_saves 75 
        --lr 0.000393141171574037 
        --ent 0.0007160293279937344 
        --cr 0.16102319952328978 
        --num_epochs 10 
        --batch_steps 65536 
        --cl

This will trigger a single training run of a closed-loop PPO for 75M
frames. On a modern CPU, this will take around 1 day of computation,
10GB memory (leak in the example environment),
and less than 2 logical cores. You can check the PNG image
with an example learning curve available in the repository as
`./resources/ppo_cl/*.png` to see what to expect over time. The results
with the log file will be available under `./results` in the same
directory as the SIMG file. To display help of the main executable through the SIMG file:

        singularity run --app help *.simg

Inside of `./results` directory there will be `./tensorboard` directory
created with episode rewards logs. Tensorboard utility that is installed
with Python Tensorflow @Abadi2016 can spawn a web server,
which is able to visualize those logs at runtime by simply pointing to
the mentioned directory:

        tensorboard --logdir tensorboard --port 6080

Upon starting the server, weblink will be displayed in the output to
render the visualization in a browser.\
\
You can of course change passed parameters, however, if you wish to
change the graph structure you will need to regenerate the graph file (MLP):

    git clone https://gitlab.doc.ic.ac.uk/sb5817/stable-baselines.git

    cd stable-baselines/

    python3 ./stable_baselines/ppo2/graph_generator.py 
        [4,5] 
        --observation_space_size 18 
        --save_path graphs/ppo_cpp_[4_5]_lr_0.0004_cr_0.1610_ent_0.0007.meta.txt 
        --learning_rate 0.000393141171574037
        --ent_coef 0.0007160293279937344
        --cliprange 0.16102319952328978

This will generate a closed-loop graph similar to the one used in the
training initiated above. Generator will write the file with respect to
your Stable Baselines @Hill2018 repository. You need to point to
this file when calling into the PPO SIMG file. In order to see what
parameters are accepted, from *within* the repository call:

    python3 ./stable_baselines/ppo2/graph_generator.py --help

If you require policy other than MLP, modifications to both graph_generator and
core PPO_CPP may be needed, however as long as the Policy is originally supported
by Stable Baselines those changes shouldn't be too challenging. The reason for
forking

Visualizing gaits
-----------------

To start the PPO gait visualization you need to [install
Singularity](https://sylabs.io/guides/3.3/user-guide/quick_start.html#quick-installation-steps)
@Kurtzer2016 version 3.3 or later (at the moment available only for
Linux systems). For your convenience, a well-performing PPO setup was
committed in the PPO repository. The following assumes you are **not**
running in the headless mode. Paying attention to the very long argument
list to the SIMG file, type in bash:

    git clone https://gitlab.doc.ic.ac.uk/sb5817/ppo_cpp.git

    cd ppo_cpp/singularity

    ./start_container.sh

    cd /git/sferes2/
    ./waf --exp ppo_cpp
    ./build/exp/ppo_cpp/ppo_cpp 
        --cl 
        -p exp/ppo_cpp/resources/ppo_cl/2019-08-20_21_13_01_2859_0.pkl.71

This will trigger a window in which hexapod will be visualized in
5-second sessions, looping forever. Close through the Ctrl+C key
combination. If you close the window manually, the simulation will just
run headless, just like in the training process. You can use `–help` to
see the full listing of available options. Serialization files are
created during the training process under `./checkpoints` directory
nested under `./results` with a name according to the chosen frame
interval (typically 1 save per 1M frames).\
\
If you are running in a headless mode, you can try running
`visu_server.sh` from any directory within the container (preferably as
a background process using &) to start a VNC @VNC server. This will bind
to the localhost on port 6080 of the host machine, where visualization
will be rendered. As a practitioner’s note, it is advisable to check if
VNC started correctly and restart it if it did not. It is also not
recommended to do this when not in headless mode due to the deep
integration of Singularity @Kurtzer2016 with the host machine that can
result in undesirable side effects.

<a name="repos"></a>Related repositories
----------------------------
-   [docker-pydart2\_hexapod\_baselines](https://gitlab.doc.ic.ac.uk/sb5817/docker-dart-gym) - Docker @Merkel2014 setup of a Python-based hexapod simulation
    environment.

-   [stable\_baselines](https://gitlab.doc.ic.ac.uk/sb5817/stable-baselines) - Fork of Stable Baselines @Hill2018 (deep RL algorithm
    suite). Includes modified PPO2 algorithm @Schulman2017 and utilities to export
    Tensorflow @Abadi2016 meta graph.

-   [gym-dart\_env](https://gitlab.doc.ic.ac.uk/sb5817/dart_env) -
    Hexapod setup as a Python-based environment within OpenAI Gym
    @Brockman2016 framework.
    
-   [pydart2](https://gitlab.doc.ic.ac.uk/sb5817/pydart2) - Fork of
    Pydart2 @Ha2016: Python layer over C++-based DART @Lee2018
    simulation framework. Modified to enable experiments with hexapod.
    
References
==========
1. Martin Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, JeffreyDean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A system for large-scale machine learning. In12th{USENIX}Sym-posium on Operating Systems Design and Implementation ({OSDI}16), pages265–283, 2016
2. Greg  Brockman,  Vicki  Cheung,  Ludwig  Pettersson,  Jonas  Schneider,  John Schulman,  Jie  Tang,  and  Wojciech Zaremba.   Openai  gym.arXiv  preprintarXiv:1606.01540, 2016
3. Antoine Cully, Jeff Clune, Danesh Tarapore, and Jean-Baptiste Mouret. Robots that can adapt like animals. Nature, 521(7553):503, 2015.
4. Gael Guennebaud, Benoit Jacob, et al.  Eigen v3. http://eigen.tuxfamily.org, 2010
5. Sehoon  Ha.Pydart2:   A  python  binding  of  DART. https://github.com/sehoonha/pydart2, 2016
6. Ashley Hill, Antonin Raffin, Maximilian Ernestus, Adam Gleave, Rene Traore, Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plap-pert,  Alec Radford,  John Schulman,  Szymon Sidor,  and Yuhuai Wu.   Stablebaselines.https://github.com/hill-a/stable-baselines, 2018
7. Gregory M Kurtzer. Singularity 2.1.2 - Linux application and environment con-tainers for science, August 2016
8. Jeongseok Lee, Michael Grey, Sehoon Ha, Tobias Kunz, Sumit Jain, Yuting Ye, Siddhartha Srinivasa, Mike Stilman, and C Karen Liu.  Dart:  Dynamic anima-tion and robotics toolkit.The Journal of Open Source Software, 3:500, 02 2018
9. Dirk Merkel. Docker: Lightweight linux containers for consistent developmentand deployment. Linux J., 2014(239), March 2014
10. Jean-Baptiste  Mouret  and  Stephane  Doncieux.   SFERESv2:   Evolvin’  in  themulti-core  world.   InProc.  of  Congress  on  Evolutionary  Computation  (CEC),pages 4079–4086, 2010
11. Thomas Nagy.The WAF Book. 2010
