//
// Created by szymon on 18/08/19.
//

#ifndef PPO_CPP_HEXAPOD_CLOSED_LOOP_ENV_HPP
#define PPO_CPP_HEXAPOD_CLOSED_LOOP_ENV_HPP

#include "hexapod_env.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

class  HexapodClosedLoopEnv : public HexapodEnv {
public:
    explicit HexapodClosedLoopEnv(double reset_noise_scale = 0.1, bool observe_velocities = false, int num_envs = 1,
                         float step_duration = 0.015,
                         float simulation_duration = 5, float min_action_value = -1, float max_action_value = 1) :
            HexapodEnv(num_envs, step_duration, simulation_duration, min_action_value, max_action_value, false),
            _reset_noise_scale{reset_noise_scale},
            observe_velocities{observe_velocities},
            observation_space_size{observe_velocities ? 36 : 18}{

        reset();
    }

    int get_observation_space_size() override {
        return observation_space_size;
    }

    Mat reset() override {
        HexapodEnv::reset();
        return std::move(apply_noise());
    }

    void serialize(nlohmann::json& json) override {
        HexapodEnv::serialize(json);
        json["reset_noise_scale"] = _reset_noise_scale;
    }

    void deserialize(nlohmann::json& json) override {
        HexapodEnv::deserialize(json);
        _reset_noise_scale = json["reset_noise_scale"].get<double>();
        apply_noise();
    }

protected:
    Mat get_obs() override {
        Mat obs{Mat(get_num_envs(), get_observation_space_size())};

        //for each environment
        obs.block(0, 0, 1, 18) = local_robot->skeleton()->getPositions().tail(18).cast<float>().transpose();

        if (observe_velocities) {
            obs.block(0, 18, 1, 18) = local_robot->skeleton()->getVelocities().tail(18).cast<float>().transpose();
        }

        return std::move(obs);
    }

private:
    double _reset_noise_scale;
    bool observe_velocities;
    int observation_space_size;

    Mat apply_noise() {
        Eigen::VectorXd qpos{local_robot->skeleton()->getPositions()};
        Eigen::VectorXd qvel{local_robot->skeleton()->getVelocities()};

        qpos.tail(18) += _reset_noise_scale * Eigen::VectorXd::Random(18);
        qvel.tail(18) += _reset_noise_scale * Eigen::VectorXd::Random(18);

        local_robot->skeleton()->setPositions(qpos);
        local_robot->skeleton()->setVelocities(qvel);

        Mat obs{get_obs()};

        old_obs = obs;

        return std::move(obs);
    }
};

#endif //PPO_CPP_HEXAPOD_CLOSED_LOOP_ENV_HPP
