//
// Created by szymon on 17/07/19.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//

#ifndef PPO_CPP_RUNNING_STATISTICS_HPP
#define PPO_CPP_RUNNING_STATISTICS_HPP

#include <Eigen/Dense>
#include "../json.hpp"
#include "serializable.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class RunningStatistics : public virtual ISerializable{
public:
    explicit RunningStatistics(int space_size = 1, float epsilon=1e-6)
    : mean{Mat::Zero(1,space_size)}
    , var{Mat::Ones(1,space_size)}
    , count{epsilon}
    , space_size{space_size}
    {

    }

    void update(const Mat& batch){
        assert(batch.cols() == space_size);

        int batch_count = batch.rows();

        const Mat& batch_mean = RunningStatistics::get_mean(batch);
        const Mat& batch_variance = RunningStatistics::get_variance(batch, batch_mean);

        update_from_moments(batch_mean, batch_variance, batch_count);
    }

    static Mat get_mean(const Mat& batch){
        Mat batch_mean = batch.colwise().mean();
        return batch_mean;
    }

    static Mat get_m2(const Mat& batch, const Mat& batch_mean){
        //assert(batch_mean.rows() == 1 && batch_mean.cols() == space_size);
        int batch_count = batch.rows();

        Mat mean_v_stacked = Eigen::VectorXf::Ones(batch_count) * batch_mean;
        Mat batch_sub_mean {batch - mean_v_stacked};
        return (batch_sub_mean.cwiseProduct(batch_sub_mean)).colwise().sum();
    }

    static Mat get_variance (const Mat& batch, const Mat& batch_mean){
        int batch_count = batch.rows();
        return get_m2(batch,batch_mean)/ static_cast<double>(batch_count);
    }

//    static RowVector get_sample_variance (const Mat& batch, const RowVector& batch_mean){
//        int batch_count = batch.rows();
//        return get_m2(batch,batch_mean)/ static_cast<double>(batch_count-1);
//    }

    void serialize(nlohmann::json& json) override {
        json["var"] = std::move(std::vector<float>(var.data(), var.data() + var.rows() * var.cols()));
        json["mean"] = std::move(std::vector<float>(mean.data(), mean.data() + mean.rows() * mean.cols()));
    }

    void deserialize(nlohmann::json& json) override {

        auto var_v {json["var"].get<std::vector<float>>()};
        auto mean_v {json["mean"].get<std::vector<float>>()};
//
//        std::cout << "var" << var_v[0] << std::endl;
//        std::cout << "mean" << mean_v[0] << std::endl;

        float* ptr = &(var_v[0]);
        var = Eigen::Map<Mat>(ptr, 1, var.cols());

        float* ptr2 = &(mean_v[0]);
        mean = Eigen::Map<Mat>(ptr2, 1, mean.cols());
//
//        std::cout << "var" <<var << std::endl;
//        std::cout << "mean" <<mean << std::endl;
    }

private:
    void update_from_moments(const Mat& batch_mean, const Mat& batch_var, double batch_count){

        Mat delta = batch_mean - mean;

        double total_count = count + batch_count;

        mean = mean + delta * batch_count/total_count;

        //m2's from variances
        Mat m_a = var * count;
        Mat m_b = batch_var * batch_count;

        Mat m_2 = m_a + m_b + delta.cwiseProduct(delta) * count * batch_count / total_count;
        var = m_2 / total_count;

        count = batch_count + count;
    }


public:
    Mat mean;
    Mat var;
    double count;
private:
    int space_size;
};


#endif //PPO_CPP_RUNNING_STATISTICS_HPP
