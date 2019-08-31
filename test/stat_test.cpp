//
// Created by szymon on 17/07/19.
//



#include <iostream>
#include "../common/running_statistics.hpp"
#include "../common/matrix_clamp.hpp"
#include "logger.hpp"
#include "catch.hpp"
#include "../common/median.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::RowVectorXf RowVector;


Mat f1(){
    Mat m {1,1};
    m(0,0) = 7;
    return std::move(m);
}
Mat f2(){
    const Mat& m = f1();
    return std::move(m);
}
//
//
//int main(){
//
//    //test2
//    float clip_obs = 5;
//    int flattened_obs_size = 5*3;
//    Mat obs_clamp_matrix {Mat::Zero(flattened_obs_size,3)};
//    obs_clamp_matrix.block(0,0,flattened_obs_size,1) = Eigen::VectorXf::Ones(flattened_obs_size) * -clip_obs;
//    obs_clamp_matrix.block(0,2,flattened_obs_size,1) = Eigen::VectorXf::Ones(flattened_obs_size) * clip_obs;
//
//
//
//    std::cout << obs_clamp_matrix.cwiseProduct(obs_clamp_matrix) << std::endl;
//
//    //test3
//    auto obs {m1};
//    Mat o = (obs.rowwise()-mean)*(var + 1e-8*RowVector::Ones(obs.cols())).cwiseSqrt().cwiseInverse().asDiagonal();
//
//    std::cout << "o" << o << std::endl;
//    Mat p = std::move(o);
//
//    std::cout << "o" << o << std::endl;
//    std::cout << "p" << p << std::endl;
//
//    //test4
//
//    auto clamp {MatrixClamp{p,-.1f,.1f}};
//
//    std::cout << clamp.clamp(p) << std::endl;
//
//    //test 5
//
//
//    //test 6
//    Logger l {"oh"};
//
//    auto& ref_l{l};
//    l = Logger("oh2");
//    std::cout << "r value oh2 died and its copy will die soon?" << std::endl;
//
//    std::cout << "oh got transformed into:" << ref_l._name << std::endl;
//
//    return 0;
//}
TEST_CASE( "Cascading const ref", "[Understanding]" )
{
    const Mat& m = f2();

    std::cout << "cascading const ref lifetime check" << std::endl;

    std::cout << m << std::endl;
}

TEST_CASE("Check running stats","[Project]")
{
    Mat m1 {4,3};

    m1 << 1,2,3,4,2,6,7,8,1,10,11,12;

    //std::cout << m1 << std::endl;

    auto mean = RunningStatistics::get_mean(m1);
    auto var = RunningStatistics::get_variance(m1, mean);
//    auto sample_var = RunningStatistics::get_sample_variance(m1, mean);

    REQUIRE((mean-Eigen::RowVector3f(5.5,5.75,5.5)).cwiseAbs().sum() < 1e-3);
    REQUIRE((var-Eigen::RowVector3f(11.25,15.1875,17.25)).cwiseAbs().sum() < 1e-3);
//    assert((sample_var-Eigen::RowVector3f(15,20.25,23)).cwiseAbs().sum() < 1e-3);

    std::cout << mean << " " << var <</*" "<< sample_var<<*/ std::endl;

    RunningStatistics rs {static_cast<int>(m1.cols()),1e-6};

    for(int i =0;i<m1.rows();++i){
        rs.update(m1.row(i));
    }

    std::cout << rs.mean << "] [" << rs.var <<"] "<< std::endl;

    REQUIRE((mean-rs.mean).cwiseAbs().sum() < 1e-3);
    REQUIRE((var-rs.var).cwiseAbs().sum() < 1e-3);
    //assert((sample_var-Eigen::RowVector3f(15,20.25,23)).cwiseAbs().sum() < 1e-3);
}

//TEST_CASE("Median","[Project]")
//{
//    Mat m1 {2,1};
//
//    m1 << 2,3;
//
//    const std::vector<float>& result = Median::median(m1,1e-4);
//
//    std::cout << result[0] << std::endl;
//
//    REQUIRE(-1e-4 < result[0]-2.5);
//    REQUIRE(result[0]-2.5 < 1e-4);
//}

//TEST_CASE( "Moving vecotr elements", "[Understanding]" )
//{
//    std::vector<std::vector<int>> parent(2);
//    parent[0] = {1,2};
//    parent[1] = {3,4};
//
//    const auto& moved = std::move(parent[1]);
//
//    std::cout << moved[0] << ',' << moved[1] << std::endl;
//
//    std::cout << parent[1][0] << ',' << parent[1][1] << std::endl;
//}

//TEST_CASE( "Vector allocators", "[Understanding]" )
//{
//    auto v {std::vector<float> (5,.5f)};
//
//    for (int i = 0; i < v.size(); ++i){
//        std::cout << v[i] << " ";
//    }
//    std::cout << std::endl;
//}


//TEST_CASE( "Random Eigen", "[Understanding]" )
//{
//
//    std::cout << 0.1f * Eigen::VectorXd::Random(18) << " ";
//
//    std::cout << std::endl;
//}

//TEST_CASE( "Column vector to std vector", "[Understanding]" )
//{
//    auto v {std::vector<float> (5,.5f)};
//
//    const Eigen::VectorXf& qpos {Eigen::VectorXf::Random(18)};
//
//    std::cout << qpos;
//
//    v.resize(qpos.size());
//    Eigen::VectorXf::Map(&v[0], qpos.size()) = qpos;
//
//    for (int i = 0; i < v.size(); ++i){
//        std::cout << v[i] << " ";
//    }
//    std::cout << std::endl;
//}