//
// Created by szymon on 10/07/19.
//

#include "ppo2/ppo2.hpp"
#include "ppo2/env.hpp"



int main(){

/*    tensorflow::TensorShape shape;
    shape.InsertDim(0, 2);

    auto t1 = tensorflow::Tensor(tensorflow::DT_FLOAT, shape).tensor<float,1>();

    shape.InsertDim(1, 2);
    shape.InsertDim(2, 2);

    auto t3 = tensorflow::Tensor(tensorflow::DT_FLOAT, shape).tensor<float,3>();

    Eigen::Tensor<double, 4, Eigen::RowMajor> t(10, 10, 10, 10);
    t(0, 1, 2, 3) = 42.0;*/

    Eigen::MatrixXf m(4,4);
    m <<  1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12,
            13,14,15,16;


    Eigen::MatrixXf m2(2,2);

    m2 << -9,-8,-7,-6;

    m.block<2,2>(1,1) = m2;

    m2(0,0)=0;
    m2(1,1)=1;

    std::cout << "m with changed Block in the middle" << std::endl;
    std::cout << m << std::endl;

    Env e {8};
    PPO2 algorithm {"./exp/ppo_cpp/ppo2_graph-1562761538.345122.meta.txt",e};

    return 0;
}
