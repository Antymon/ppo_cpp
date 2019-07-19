//
// Created by szymon on 17/07/19.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//

#ifndef PPO_CPP_RUNNING_STATISTICS_HPP
#define PPO_CPP_RUNNING_STATISTICS_HPP

#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::RowVectorXf RowVector;

class RunningStatistics{
public:
    explicit RunningStatistics(int space_size = 1, float epsilon=1e-6)
    : mean{RowVector::Zero(space_size)}
    , var{RowVector::Ones(space_size)}
    , count{epsilon}
    , space_size{space_size}
    {

    }

    void update(const Mat& batch){
        assert(batch.cols() == space_size);

        int batch_count = batch.rows();

        const RowVector& batch_mean = RunningStatistics::get_mean(batch);
        const RowVector& batch_variance = RunningStatistics::get_variance(batch, batch_mean);

        update_from_moments(batch_mean, batch_variance, batch_count);
    }

    static RowVector get_mean(const Mat& batch){
        RowVector batch_mean = batch.colwise().mean();
        return batch_mean;
    }

    static RowVector get_m2(const Mat& batch, const RowVector& batch_mean){
        //assert(batch_mean.rows() == 1 && batch_mean.cols() == space_size);
        int batch_count = batch.rows();

        Mat mean_v_stacked = Eigen::VectorXf::Ones(batch_count) * batch_mean;
        Mat batch_sub_mean {batch - mean_v_stacked};
        return (batch_sub_mean.cwiseProduct(batch_sub_mean)).colwise().sum();
    }

    static RowVector get_variance (const Mat& batch, const RowVector& batch_mean){
        int batch_count = batch.rows();
        return get_m2(batch,batch_mean)/ static_cast<double>(batch_count);
    }

//    static RowVector get_sample_variance (const Mat& batch, const RowVector& batch_mean){
//        int batch_count = batch.rows();
//        return get_m2(batch,batch_mean)/ static_cast<double>(batch_count-1);
//    }

private:
    void update_from_moments(const RowVector& batch_mean, const RowVector& batch_var, double batch_count){

        RowVector delta = batch_mean - mean;

        double total_count = count + batch_count;

        mean = mean + delta * batch_count/total_count;

        //m2's from variances
        RowVector m_a = var * count;
        RowVector m_b = batch_var * batch_count;

        RowVector m_2 = m_a + m_b + delta.cwiseProduct(delta) * count * batch_count / total_count;
        var = m_2 / total_count;

        count = batch_count + count;
    }


public:
    RowVector mean;
    RowVector var;
    double count;
private:
    int space_size;
};


#endif //PPO_CPP_RUNNING_STATISTICS_HPP
