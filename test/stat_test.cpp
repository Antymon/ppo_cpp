//
// Created by szymon on 17/07/19.
//



#include <iostream>
#include "../common/running_statistics.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::RowVectorXf RowVector;

int main(){

    Mat m1 {4,3};

    m1 << 1,2,3,4,2,6,7,8,1,10,11,12;

    //std::cout << m1 << std::endl;

    auto mean = RunningStatistics::get_mean(m1);
    auto var = RunningStatistics::get_variance(m1, mean);
    auto sample_var = RunningStatistics::get_sample_variance(m1, mean);

    assert((mean-Eigen::RowVector3f(5.5,5.75,5.5)).cwiseAbs().sum() < 1e-3);
    assert((var-Eigen::RowVector3f(11.25,15.1875,17.25)).cwiseAbs().sum() < 1e-3);
    assert((sample_var-Eigen::RowVector3f(15,20.25,23)).cwiseAbs().sum() < 1e-3);

    std::cout << mean << " " << var <<" "<< sample_var<< std::endl;

    RunningStatistics rs {static_cast<int>(m1.cols()),1e-6};

    for(int i =0;i<m1.rows();++i){
        rs.update(m1.row(i));
    }

    std::cout << rs.mean << "] [" << rs.var <<"] "<< std::endl;

    assert((mean-rs.mean).cwiseAbs().sum() < 1e-3);
    assert((var-rs.var).cwiseAbs().sum() < 1e-3);
    //assert((sample_var-Eigen::RowVector3f(15,20.25,23)).cwiseAbs().sum() < 1e-3);

    return 0;
}
