//
// Created by szymon on 13/07/19.
//

#include <memory>
#include <Eigen/Dense>
#include <string>
#include <iostream>


typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

int main(){
    Mat a {2,2};
    a << 1,2,3,4;
    Mat b = a;
    a << 5,6,7,8;
    std::cout << b << std::endl;
}