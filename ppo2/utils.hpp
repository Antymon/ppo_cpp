//
// Created by szymon on 11/07/19.
//

#ifndef PPO_CPP_UTILS_HPP
#define PPO_CPP_UTILS_HPP

#include <tensorflow/core/framework/tensor.h>
#include "tensorboard.hpp"
#include <algorithm>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;


class Utils {
public:
    static void convert_tensor(tensorflow::Tensor &t, Mat &m, std::string space_type = Env::SPACE_CONTINOUS) {
        if (space_type==Env::SPACE_DISCRETE){
            return convert_tensor<tensorflow::int64>(t,m);
        } else {
            return convert_tensor<float>(t,m);
        }
    }


    template<typename T>
    static void convert_tensor(tensorflow::Tensor &t, Mat &m) {
        assert(t.dims() < 3); //not suported
        switch (t.dims()) {
            case 0:
                m = Mat(1, 1); //scalar
                break;
            case 1:
                m = Mat(t.dim_size(0), 1);
                break;
            case 2:
                m = Mat(t.dim_size(0), t.dim_size(1));
                break;
        }

        auto src = t.flat<T>().data();
        memcpy(m.data(), src, m.cols() * m.rows() * sizeof(T));
    }


    static void convert_mat(const Mat &m, tensorflow::Tensor &t) {
        tensorflow::TensorShape shape;
        shape.InsertDim(0, m.rows());
        shape.InsertDim(1, m.cols());

        t = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
        auto dst = t.flat<float>().data();
        memcpy(dst, m.data(), m.cols() * m.rows() * sizeof(float));

    }

    static void convert_vec(const Mat &m, tensorflow::Tensor &t) {
        assert(m.cols() == 1);

        tensorflow::TensorShape shape;
        shape.InsertDim(0, m.rows());

        t = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
        auto dst = t.flat<float>().data();
        memcpy(dst, m.data(), m.cols() * m.rows() * sizeof(float));

        assert(t.dims() == 1);
    }

    static void scalar(float scalar, tensorflow::Tensor &t) {
        t = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape());
        t.scalar<float>()() = scalar;
    }

    static Mat total_episode_reward_logger(Mat rew_acc, Eigen::Map<Mat> rewards_view, Eigen::Map<Mat> masks_view, TensorboardWriter& writer, int total_steps){
        assert(rew_acc.rows() == rewards_view.rows() && rewards_view.rows() == masks_view.rows());
        assert(rew_acc.cols() == 1 && rewards_view.cols() == masks_view.cols());

        const int steps = rewards_view.cols();


        for (int env_idx = 0; env_idx < rew_acc.rows(); ++env_idx){

            std::vector<int> dones_idx{};

            for (int step = 0; step < steps; ++step){
                if(masks_view(env_idx,step) > .5f){ //in theory those values should be 0. or 1. but for the sake of limited precision comparing against .5
                    dones_idx.push_back(step);
                    //std::cout<< "#9" << std::endl;
                }
            }


            if(dones_idx.empty()){
                rew_acc(env_idx,0) += rewards_view.row(env_idx).sum();
            } else {
                rew_acc(env_idx,0) += rewards_view.block(env_idx,0,1,dones_idx[0]).sum();

                //std::cout<<rew_acc(env_idx,0);
                writer.write_scalar(total_steps+dones_idx[0],"episode_reward",rew_acc(env_idx,0));

                for (int k = 1; k<dones_idx.size(); ++k){
                    rew_acc(env_idx,0) = rewards_view.block(env_idx,dones_idx[k-1],1,dones_idx[k]-dones_idx[k-1]).sum();

                    //std::cout<<rew_acc(env_idx,0);
                    writer.write_scalar(total_steps+dones_idx[k],"episode_reward",rew_acc(env_idx,0));
                }
                rew_acc(env_idx,0) = rewards_view.block(env_idx,dones_idx[dones_idx.size()-1],1,steps-dones_idx[dones_idx.size()-1]).sum();
            }

        }

        return rew_acc;
    }
};
#endif //PPO_CPP_UTILS_HPP
