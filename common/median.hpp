//
// Created by szymon on 20/08/19.
//

#ifndef PPO_CPP_MEDIAN_HPP
#define PPO_CPP_MEDIAN_HPP

#include <Eigen/Dense>

#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class Median {
public:

    /*
     * Weiszfeld's algorithm which doesn't deal with colinearity and situation when
     * estimate overlaps actual point; unless improved, not used
     */

    static std::vector<float> median(const Mat& input, float epsilon = 1e-4){

//        std::cout << input << std::endl;

        Mat y {Mat::Zero(1,input.cols())};
        Mat new_y {Mat::Zero(1,input.cols())};

        float error;

        do {
            y = new_y;

            Mat stack_y{Mat::Ones(input.rows(), 1) * y};

//            std::cout << stack_y << std::endl;

            Mat input_sub_y{input - stack_y};

//            std::cout << input_sub_y << std::endl;

            Mat input_sub_y_sq{input_sub_y.cwiseProduct(input_sub_y)};

//            std::cout << input_sub_y_sq << std::endl;

            Mat distances{input_sub_y_sq.rowwise().sum().cwiseSqrt()};

//            std::cout << distances << std::endl;

            Mat inverted_distances{Mat::Zero(input.rows(), 1)};

            for (int i = 0; i < input.rows(); ++i) {
                if (distances(i, 0) != 0.f) {
                    inverted_distances(i, 0) = 1.f / distances(i, 0);
                }
            }

//            std::cout << inverted_distances << std::endl;

            Mat input_scaled{input.cwiseProduct(inverted_distances * Mat::Ones(1, input.cols()))};

//            std::cout << input_scaled << std::endl;

            new_y = input_scaled.colwise().sum() / inverted_distances.sum();

//            std::cout << new_y << std::endl;

            error = std::sqrt((new_y-y).cwiseProduct(new_y-y).sum());

//            std::cout << "error " << error << std::endl;

        } while (error > epsilon);

        std::vector<float> result(new_y.size());

        Eigen::Map<Mat>(result.data(), new_y.rows(), new_y.cols()) = new_y;

        return std::move(result);
    }
};

#endif //PPO_CPP_MEDIAN_HPP
