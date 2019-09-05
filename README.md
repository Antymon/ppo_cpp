User Guide
==========

Overview
--------

The base setup used to develop and run elements of this project
(therefore the recommended setup) includes:

-   Ubuntu 18.04 LTS operating system,

-   Singularity @singularity containerization environment,

-   set of project repositories listed under Section [chapter~R~epo].

Training gaits
--------------

To start the PPO training you need to [install
Singularity](https://sylabs.io/guides/3.3/user-guide/quick_start.html#quick-installation-steps)
@singularity version 3.3 or later (at the moment available only for
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
10GB memory, and less than 2 logical cores. You can check the PNG image
with an example learning curve available in the repository as
`./resources/ppo_cl/*.png` to see what to expect over time. The results
with the log file will be available under `./results` in the same
directory as the SIMG file.\
\
Inside of `./results` directory there will be `./tensorboard` directory
created with episode rewards logs. Tensorboard utility that is installed
with Python Tensorflow @abadi2016tensorflow can spawn a web server,
which is able to visualize those logs at runtime by simply pointing to
the mentioned directory:

        tensorboard --logdir tensorboard --port 6080

Upon starting the server, weblink will be displayed in the output to
render the visualization in a browser.\
\
You can of course change passed parameters, however, if you wish to
change the graph structure you will need to regenerate the graph file:

    git clone https://gitlab.doc.ic.ac.uk/sb5817/stable-baselines.git

    cd stable-baselines/

    pip3 install -e .

    python3 ./stable_baselines/ppo2/graph_generator.py 
        [4,5] 
        --observation_space_size 18 
        --save_path graphs/ppo_cpp_[4_5]_lr_0.0004_cr_0.1610_ent_0.0007.meta.txt 
        --learning_rate 0.000393141171574037
        --ent_coef 0.0007160293279937344
        --cliprange 0.16102319952328978

This will generate a closed-loop graph similar to the one used in the
training initiated above. Generator will write the file with respect to
your Stable Baselines @stable-baselines repository. You need to point to
this file when calling into the PPO SIMG file. In order to see what
parameters are accepted, from *within* the repository call:

    python3 ./stable_baselines/ppo2/graph_generator.py --help

Visualizing gaits
-----------------

To start the PPO gait visualization you need to [install
Singularity](https://sylabs.io/guides/3.3/user-guide/quick_start.html#quick-installation-steps)
@singularity version 3.3 or later (at the moment available only for
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
integration of Singularity @singularity with the host machine that can
result in undesirable side effects.

Project repositories listing
----------------------------

All repositories are currently hosted under Imperial College London’s
Gitlab service, although they will be migrated to [author’s github
profile](https://github.com/Antymon) upon expiry of author’s student
account. Currently, repositories include:

-   [neural\_mape](https://gitlab.doc.ic.ac.uk/sb5817/neural-mape) -
    Representing hexapod controller as a small fully-connected neural
    network evolved with MAP Elites @mouret2015illuminating algorithm.

-   [signal\_approximation](https://gitlab.doc.ic.ac.uk/sb5817/msc) -
    Hexapod reference controller’s (as per @cully2015robots)
    approximation with a fully-connected neural network.

-   [docker-pydart2\_hexapod\_baselines](https://gitlab.doc.ic.ac.uk/sb5817/docker-dart-gym) - Docker @docker setup of a Python-based hexapod simulation
    environment.

-   [pydart2](https://gitlab.doc.ic.ac.uk/sb5817/pydart2) - Fork of
    Pydart2 @pydart: Python layer over C++-based DART @lee2018dart
    simulation framework. Modified to enable experiments with hexapod.

-   [stable\_baselines](https://gitlab.doc.ic.ac.uk/sb5817/stable-baselines) - Fork of Stable Baselines @stable-baselines (deep RL algorithm
    suite). Includes modified PPO2 algorithm and utilities to export
    Tensorflow @abadi2016tensorflow meta graph.

-   [ppo\_cpp](https://gitlab.doc.ic.ac.uk/sb5817/ppo_cpp) - C++ port of
    the training part of Stable Baselines @stable-baselines PPO2
    algorithm and utilities. Requires Tensorflow @abadi2016tensorflow
    meta graph as input.

-   [docker-neural\_mape](https://gitlab.doc.ic.ac.uk/sb5817/szymonbrych-airl_env) - Docker @docker setup of MAP Elites experiment. Not maintained due
    to migration towards Singularity @singularity.

-   [gym-dart\_env](https://gitlab.doc.ic.ac.uk/sb5817/dart_env) -
    Hexapod setup as a Python-based environment within OpenAI Gym
    @brockman2016openai framework.

-   [nn2](https://gitlab.doc.ic.ac.uk/sb5817/nn2.git) - Fork of NN2 @nn2
    plugin introducing evolvable MLP network.

-   [sferes2](https://gitlab.doc.ic.ac.uk/sb5817/sferes2) - Fork of
    Sferes2 @mouret2010sferes evolutionary framework making fixes to not
    maintained features such as resuming training.


