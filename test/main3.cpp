//
// Created by szymon on 13/07/19.
//

#include <memory>
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <random>


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

    //std::cout << *obs << std::endl;

    (*obs) = M2;



//    for (int env_id=0; env_id<obs->rows();++env_id){
//        auto row = obs->row(env_id);
//        new_mb_obs->block(env_id*n_steps,0,n_steps,obs_size) = row;
//    }
//    obs = new_mb_obs;

// test 2

    int noptepochs = 2;

    auto t = time(0);

    for (int epoch_num = 0; epoch_num<noptepochs; epoch_num++) {
        //std::random_shuffle(inds.begin(), inds.end());

        auto seed = t+2*epoch_num;

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(obs->rows());
        perm.setIdentity();
        std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), std::default_random_engine(seed));
        Mat tmp = perm * *(obs);

        std::cout << tmp << std::endl;
        std::cout << std::endl;

        //seed = t+2*epoch_num+1;

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm2(obs->rows());
        perm2.setIdentity();
        std::shuffle(perm2.indices().data(), perm2.indices().data() + perm2.indices().size(), std::default_random_engine(seed));
        Mat tmp2 = perm2 * *(obs);

        std::cout << tmp2 << std::endl;
        std::cout << std::endl;
    }

    //test 3

    std::cout << "test3"<< std::endl;

    auto obs2 = std::make_shared<Mat>(num_envs*n_steps,obs_size);

    *obs2 << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29;

    auto obs3 = std::make_shared<Mat>(num_envs*n_steps,obs_size);

    *obs3 << 0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29;

    const auto& mb_all = std::vector<std::shared_ptr<Mat>>{obs2,obs3};
    //all batch vectors have same num of rows
    auto num_rows = mb_all[0]->rows();
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm{num_rows};
    perm.setIdentity();

    std::cout << "marker 1"<< std::endl;

    for (int epoch_num = 0; epoch_num<noptepochs; epoch_num++) {
        //std::random_shuffle(inds.begin(), inds.end());

        std::vector<std::shared_ptr<Mat>> mb_permutated{mb_all.size()};

        std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());

        std::cout << "marker 2"<< std::endl;

        for (int i = 0; i < mb_permutated.size(); ++i) {
            const auto &v = mb_all[i];

            auto tmp = std::make_shared<Mat>();
            std::cout << "marker 3"<< std::endl;
            *tmp = perm * *v;

            std::cout << "marker 4"<< std::endl;
            mb_permutated[i] = tmp;
        }



        for (int i = 0; i < mb_permutated.size(); ++i) {
            std::cout << *(mb_permutated[i]) << std::endl;
        }

    }

}