//
// Created by szymon on 11/07/19.
//

#ifndef PPO_CPP_UTILS_HPP
#define PPO_CPP_UTILS_HPP

#include <tensorflow/core/framework/tensor.h>

class Utils {
public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

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

        auto src = t.flat<float>().data();
        memcpy(m.data(), src, m.cols() * m.rows() * sizeof(float));
    }


    static void convert_mat(const Mat &m, tensorflow::Tensor &t) {
        tensorflow::TensorShape shape;
        shape.InsertDim(0, m.rows());
        shape.InsertDim(1, m.cols());

        t = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
        auto dst = t.flat<float>().data();
        memcpy(dst, m.data(), m.cols() * m.rows() * sizeof(float));

    }
};
#endif //PPO_CPP_UTILS_HPP
