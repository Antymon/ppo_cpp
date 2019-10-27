#include <iostream>
#include <memory>

#include "catch.hpp"

#include "../env/env_mock.hpp"
#include "../env/vec_env.hpp"


typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

void simulate_steps(const int steps, const int num_threads){
    std::vector<std::shared_ptr<Env>> envs;

    Mat test_column {Mat::Zero(num_threads,1)};

    for (int i =0; i<num_threads; ++i){
        envs.push_back(std::make_shared<EnvMock>(i+1));
        test_column(i,0)=i+1;
    }

    VecEnv ve {envs};

    for (int i =0; i<steps; ++i){
        Mat actions {Mat::Zero(ve.get_num_envs(),ve.get_observation_space_size())};

        const auto& result = ve.step(actions);

        const Mat& obs = result[0];
        const Mat& rew = result[1];

        REQUIRE(obs.rows() == ve.get_num_envs());
        REQUIRE(rew.rows() == ve.get_num_envs());
        REQUIRE(obs.cols() == ve.get_observation_space_size());
        REQUIRE(rew.cols() == 1);

        //under assumption of mock env setup each column of obs and rews should look like test column

        REQUIRE(0.f == Approx((rew-test_column).squaredNorm()).epsilon(1e-2));

        for (int i = 0; i<ve.get_observation_space_size(); ++i){
            REQUIRE(0.f == Approx((obs.col(i)-test_column).squaredNorm()).epsilon(1e-2));
        }
    }
}


TEST_CASE( "VecEnv step results", "[Threading]" )
{
    simulate_steps(1,16);
}

