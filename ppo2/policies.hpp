//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_POLICIES_HPP
#define PPO_CPP_POLICIES_HPP

#include <Eigen/src/Core/Matrix.h>
#include <string>
#include <tensorflow/core/public/session.h>

class BasePolicy{
protected:
    static const std::string obs_ph;
};

class ActorCriticPolicy : public virtual BasePolicy{
protected:
    static const std::string action;
    static const std::string neglogp;
    static const std::string value_flat;
};

class MlpPolicy : public virtual ActorCriticPolicy{

public:

    explicit MlpPolicy(std::shared_ptr<tensorflow::Session> session):_session{session}{

    }

std::vector<tensorflow::Tensor> step(const tensorflow::Tensor& obs){

    std::vector<tensorflow::Tensor> outputs;

    tensorflow::Status s = _session->Run({{obs_ph,obs}},{action, value_flat, neglogp}, {}, &outputs);

    if (!s.ok()) {
        std::cout << "step error" << std::endl;
        std::cout << s.ToString() << std::endl;
        return {};
    }

    return outputs;
}

tensorflow::Tensor value(const tensorflow::Tensor& obs){

        std::vector<tensorflow::Tensor> outputs;

        tensorflow::Status s = _session->Run({{obs_ph,obs}},{value_flat}, {}, &outputs);

        if (!s.ok()) {
            std::cout << "value() error" << std::endl;
            std::cout << s.ToString() << std::endl;
        }

        return outputs[0];
    }

private:
    std::shared_ptr<tensorflow::Session> _session;

};

#endif //PPO_CPP_POLICIES_HPP
