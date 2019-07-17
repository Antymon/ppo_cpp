//
// Created by szymon on 17/07/19.
//

#ifndef PPO_CPP_MATRIX_CLAMP_HPP
#define PPO_CPP_MATRIX_CLAMP_HPP

#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class MatrixClamp {
public:

    MatrixClamp(const Mat& like, float clamp)
            : MatrixClamp(like.rows(),like.cols(),-clamp,clamp)
    {}

    MatrixClamp(const Mat& like, float low, float high)
            : MatrixClamp(like.rows(),like.cols(),low,high)
    {}

    MatrixClamp(int rows, int cols,  float clamp)
            : MatrixClamp(rows,cols,-clamp,clamp)
    {}

    MatrixClamp(int rows, int cols,  float low, float high)
            : lo {Mat::Ones(rows,cols)*low}
            , hi {Mat::Ones(rows,cols)*high}
    {}

    Mat clamp(const Mat& mat) const{
        Mat result = mat.cwiseMax(lo).cwiseMin(hi);
        return std::move(result);
    }

private:
    Mat lo;
    Mat hi;
};


#endif //PPO_CPP_MATRIX_CLAMP_HPP
