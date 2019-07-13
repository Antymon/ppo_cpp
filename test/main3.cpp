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


    int num_envs = 2;
    int n_steps = 3;
    int obs_size = 5;

    auto obs = std::make_shared<Mat>(num_envs,n_steps*obs_size);

    *obs << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29;

    Eigen::Map<Mat> M2(obs->data(), num_envs*n_steps,obs_size);

    std::cout << *obs << std::endl;

    (*obs) = M2;

    std::cout << *obs << std::endl;

//    for (int env_id=0; env_id<obs->rows();++env_id){
//        auto row = obs->row(env_id);
//        new_mb_obs->block(env_id*n_steps,0,n_steps,obs_size) = row;
//    }
//    obs = new_mb_obs;
}