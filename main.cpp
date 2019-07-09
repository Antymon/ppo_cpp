#include <iostream>
#include <Eigen/Dense>

#include <vector>

//#include <tensorflow/cc/client/client_session.h>
//#include <tensorflow/cc/framework/scope.h>
//#include <tensorflow/cc/ops/math_ops.h>
//#include <tensorflow/cc/ops/array_ops.h>

int main()
{
//    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
//    auto a = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
//    auto c = tensorflow::ops::Add(root, a, {41});
//
//    tensorflow::ClientSession session(root);
//    std::vector<tensorflow::Tensor> outputs;
//
//    tensorflow::Status s = session.Run({ {a, {1}} }, {c}, &outputs);
//    if (!s.ok()) {

        const int n_steps{3};
        const int n_envs{2};
        const float gamma = 0.99;
        const float lam = 0.95;

        Eigen::MatrixXf mb_rewards{n_steps, n_envs};
        Eigen::MatrixXf mb_dones{n_steps, n_envs};
        Eigen::MatrixXf mb_advs{n_steps, n_envs};
        Eigen::MatrixXf mb_returns{n_steps, n_envs};
        Eigen::MatrixXf mb_values{n_steps, n_envs};

        Eigen::MatrixXf dones{1, n_envs};
        Eigen::MatrixXf last_values{1, n_envs};

        Eigen::MatrixXf nextnonterminal{1, n_envs};
        Eigen::MatrixXf nextvalues{1, n_envs};
        Eigen::MatrixXf delta{1, n_envs};

//    mb_dones << 0.f,0.f,0.f,0.f,0.f,0.f;
//    mb_rewards << -1.18242046f, -0.43078828f,-0.12466504f, -0.85010344f,0.83386647f, -0.03693804f;
//    mb_values << -0.05811596, -0.36265668, 1.22605811, -0.42979062, 0.81316504, -0.71250895;
//    last_values << -1.76390123,  1.70628349;

        Eigen::MatrixXf last_gae_lam{Eigen::MatrixXf::Zero(1, n_envs)};

        for (int step = n_steps - 1; step >= 0; --step) {
            if (step == n_steps - 1) {

                nextnonterminal = Eigen::MatrixXf::Ones(1, n_envs) - dones;
                nextvalues = last_values;
            } else {
                nextnonterminal = Eigen::MatrixXf::Ones(1, n_envs) - mb_dones(step + 1, Eigen::seqN(0, n_envs));
                nextvalues = mb_values(step + 1, Eigen::seqN(0, n_envs));
            }
            delta = mb_rewards(step, Eigen::seqN(0, n_envs)) + gamma * nextvalues.cwiseProduct(nextnonterminal) -
                    mb_values(step, Eigen::seqN(0, n_envs));
            mb_advs(step, Eigen::seqN(0, n_envs)) = last_gae_lam =
                    delta + gamma * lam * nextnonterminal.cwiseProduct(last_gae_lam);
        }
        mb_returns = mb_advs + mb_values;

//    Eigen::MatrixXf mb_returns_check{n_steps,n_envs};
//    mb_returns_check << -2.00817212,  0.17675461, -0.94252157, 0.66859918,-0.91239575, 1.65228262;
//    std::cout << (((mb_returns-mb_returns_check).sum())<1e-5) << std::endl;
//    }
}