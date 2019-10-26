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
typedef Eigen::RowVectorXf RowVector;

TEST_CASE( "VecEnv lifetime", "[Threading]" )
{
    auto em = std::make_shared<EnvMock>(1);
    std::vector<std::shared_ptr<Env>> envs;
    envs.push_back(em);

    VecEnv ve {envs};
}

