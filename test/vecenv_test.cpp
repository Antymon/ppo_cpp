#include <iostream>
#include "../common/running_statistics.hpp"
#include "../common/matrix_clamp.hpp"
#include "logger.hpp"
#include "catch.hpp"
#include "../common/median.hpp"

#include "../env/env_mock.hpp"
#include "../env/vec_env.hpp"

#include <memory>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

TEST_CASE( "VecEnv lifetime", "[Threading]" )
{
    const int num_threads = 1;

    std::vector<std::shared_ptr<Env>> envs;
    for (int i =0; i<num_threads; ++i){
        envs.push_back(std::make_shared<EnvMock>(i+1));
    }

    VecEnv ve {envs};

    Mat actions {Mat::Zero(ve.get_num_envs(),ve.get_observation_space_size())};

    ve.step(actions);
}

