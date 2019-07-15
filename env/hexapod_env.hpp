//
// Created by szymon on 15/07/19.
//

#ifndef PPO_CPP_HEXAPOD_ENV_HPP
#define PPO_CPP_HEXAPOD_ENV_HPP

#include "../ppo2/utils.hpp"
#include <algorithm>
#include <robot_dart/robot_dart_simu.hpp>
#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/constraint/ConstraintSolver.hpp>

#include "env.hpp"


//initialising a global variable (but using a namespace) - this global variable is the robot object
namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot; //initialize a shared pointer to the object Robot from robotdart and name it global_robot
}

//Function to initialise the hexapod robot
void load_and_init_robot() {
    std::cout << "INIT Robot" << std::endl; //Print INIT Robot
    //create a shared pointer to define the global_robot object as the hexapod_v2 robot from the file
    global::global_robot = std::make_shared<robot_dart::Robot>("./exp/ppo_cpp/resources/hexapod_v2.urdf");
    global::global_robot->set_position_enforced(true);
    global::global_robot->set_actuator_types(dart::dynamics::Joint::SERVO);
    global::global_robot->skeleton()->enableSelfCollisionCheck();
    std::cout << "End init Robot" << std::endl; //Print End init Robot
}

class HexapodEnv : public virtual Env
{
public:
    HexapodEnv(int num_envs, float step_duration = 0.015, float simulation_duration = 5, float min_action_value = -1, float max_action_value = 1):
        Env(num_envs),
        step_duration{step_duration},
        simulation_duration{simulation_duration},
        simulation{step_duration},
        min_action_value{min_action_value},
        max_action_value{max_action_value}
    {
        simulation.world()->getConstraintSolver()->setCollisionDetector(
                dart::collision::BulletCollisionDetector::create());

        simulation.add_floor();

        reset();
    }

    std::string get_action_space() override {
        return Env::SPACE_CONTINOUS;
    }

    std::string get_observation_space() override {
        return Env::SPACE_CONTINOUS;
    }

    int get_action_space_size() override{
        return action_space_size;
    }

    int get_observation_space_size() override{
        return observation_space_size;
    }

    Mat reset() override {

        simulation.clear_robots();
        local_robot.reset();
        local_robot = global::global_robot->clone();
        local_robot->skeleton()->setPosition(5, 0.15);
        simulation.add_robot(local_robot);
        simulation.world()->setTime(0);

        return Mat::Zero(1,get_observation_space_size());
    }

    std::vector<Mat> step(const Mat &actions) override {

        assert(actions.cols() == get_action_space_size());

        auto pos_before = local_robot->skeleton()->getPositions().head(6).tail(3).transpose();

        float t = simulation.world()->getTime();

/*            std::cout << index << std::endl;
            for (int i = 0; i<18; i++) {
                std::cout << angles2[i+1]<< (i==17?"":", ");
            }
            std::cout << std::endl;*/

        Eigen::VectorXd target_positions = Eigen::VectorXd::Zero(action_space_size + 6);
        for (size_t i = 0; i < action_space_size; i++)
            target_positions(i + 6) = ((i % 3 == 1) ? 1.0 : -1.0) * Utils::clamp(actions(0,i),min_action_value,max_action_value);

        Eigen::VectorXd q = local_robot->skeleton()->getPositions();
        Eigen::VectorXd q_err = target_positions - q;

        double gain = 1.0 / (dart::math::constants<double>::pi() * step_duration);

        Eigen::VectorXd commands = q_err * gain;

        commands.head(6) = Eigen::VectorXd::Zero(6);

        local_robot->skeleton()->setCommands(commands);

        local_robot->update(t);
        simulation.world()->step(false);

        auto pos_after = local_robot->skeleton()->getPositions().head(6).tail(3).transpose();

        auto s = (pos_after - pos_before);

//        std::cout << "distance by axis:" << s << std::endl;
//        std::cout << "total distance:" << s.norm() << std::endl;
//        std::cout << "velocity by axis:" << s/duration.Get() << std::endl;


        Mat rewards {Mat(1,1)};
        rewards(0,0) = s[0];

        Mat dones {Mat(1,1)};
        dones(0,0) = t>=simulation_duration?1.f:0.f;

        Mat obs {Mat(1,1)};
        obs(0,0) = t;

        return {obs,rewards,dones};
    }

private:
    static const int action_space_size = 18;
    static const int observation_space_size = 1;

    float step_duration;
    float simulation_duration;
    robot_dart::RobotDARTSimu simulation;
    std::shared_ptr<robot_dart::Robot> local_robot;
    float min_action_value;
    float max_action_value;
};

#endif //PPO_CPP_HEXAPOD_ENV_HPP
