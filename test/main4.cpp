//
// Created by szymon on 14/07/19.
//
#include <Eigen/Dense>
#include <iostream>
#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

Mat total_episode_reward_logger(Mat rew_acc, Eigen::Map<Mat> rewards_view, Eigen::Map<Mat> masks_view,/*const TensorboardWriter& writer,*/ int steps){
    assert(rew_acc.rows() == rewards_view.rows() && rewards_view.rows() == masks_view.rows());
    assert(rew_acc.cols() == 1 && rewards_view.cols() == masks_view.cols() && rewards_view.cols() == steps);

    for (int env_idx = 0; env_idx < rew_acc.rows(); ++env_idx){

        std::vector<int> dones_idx{};

        for (int step = 0; step < steps; ++step){
            if(masks_view(env_idx,step) == 1.){
                dones_idx.push_back(step);
            }
        }

        if(dones_idx.size() == 0){
            rew_acc(env_idx,0) += rewards_view.row(env_idx).sum();
        } else {
            rew_acc(env_idx,0) += rewards_view.block(env_idx,0,1,dones_idx[0]).sum();

            std::cout<<rew_acc(env_idx,0);

            for (int k = 1; k<dones_idx.size(); ++k){
                rew_acc(env_idx,0) = rewards_view.block(env_idx,dones_idx[k-1],1,dones_idx[k]-dones_idx[k-1]).sum();
                std::cout<<rew_acc(env_idx,0);
            }
            rew_acc(env_idx,0) = rewards_view.block(env_idx,dones_idx[dones_idx.size()-1],1,steps-dones_idx[dones_idx.size()-1]).sum();
        }

    }

    return rew_acc;
}

int main(){

    Mat rewards {Mat::Random(2,5)};
    Mat dones {Mat::Zero(2,5)};
    Mat reward_acc {Mat::Zero(2,1)};

    std::cout << rewards << std::endl;

    int steps=5;

    dones(0,1)=1.;
    dones(0,3)=1.;

    std::cout << dones << std::endl;

    Eigen::Map<Mat> rewards_view(rewards.data(), rewards.rows(), rewards.cols());
    Eigen::Map<Mat> dones_view(dones.data(), dones.rows(), dones.cols());

    reward_acc = total_episode_reward_logger(reward_acc,rewards_view,dones_view,steps);

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << reward_acc << std::endl;

    Mat expected_reward_acc {Mat::Zero(2,1)};

    expected_reward_acc << 1.420175, -0.734503;

    std::cout << ((expected_reward_acc-reward_acc).cwiseAbs().sum());
}