//
// Created by szymon on 15/07/19.
//

#ifndef PPO_CPP_HEXAPOD_ENV_HPP
#define PPO_CPP_HEXAPOD_ENV_HPP

#ifdef GRAPHIC
#include <robot_dart/graphics/graphics.hpp>
#endif

#include <algorithm>
#include <robot_dart/robot_dart_simu.hpp>
#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/constraint/ConstraintSolver.hpp>
#include <robot_dart/control/hexa_control.hpp>

#include "env.hpp"
#include "../common/matrix_clamp.hpp"


//initialising a global variable (but using a namespace) - this global variable is the robot object
namespace global2 {
    std::shared_ptr<robot_dart::Robot> global_robot; //initialize a shared pointer to the object Robot from robotdart and name it global_robot
}

//Function to initialise the hexapod robot
void load_and_init_robot2() {
    std::cout << "INIT Robot" << std::endl; //Print INIT Robot
    //create a shared pointer to define the global_robot object as the hexapod_v2 robot from the file
    global2::global_robot = std::make_shared<robot_dart::Robot>("./exp/ppo_cpp/resources/hexapod_v2.urdf");
    global2::global_robot->set_position_enforced(true);
    global2::global_robot->set_actuator_types(dart::dynamics::Joint::SERVO);
    global2::global_robot->skeleton()->enableSelfCollisionCheck();
    std::cout << "End init Robot" << std::endl; //Print End init Robot
}

class HexapodEnv : public virtual Env
{
public:
    HexapodEnv(int num_envs, float step_duration = 0.015, float simulation_duration = 5, float min_action_value = -1, float max_action_value = 1, bool init_reset = true):
        Env(),
        step_duration{step_duration},
        simulation_duration{simulation_duration},
        simulation{step_duration},
        min_action_value{min_action_value},
        max_action_value{max_action_value},
        reward_accumulator{0},
        old_obs{},
        old_rew{Mat::Zero(get_num_envs(),1)}
        //actions_clamp{1,action_space_size,min_action_value,max_action_value}
    {

#ifdef GRAPHIC
        simulation.set_graphics(std::make_shared<robot_dart::graphics::Graphics>(simulation.world()));
        std::static_pointer_cast<robot_dart::graphics::Graphics>(simulation.graphics())->look_at({0.5, 3., 0.75}, {0.5, 0., 0.2});
#endif
        simulation.world()->getConstraintSolver()->setCollisionDetector(
                dart::collision::BulletCollisionDetector::create());

        simulation.add_floor();

        if(init_reset) {
            reset();
        }
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

        //std::cout << "reset(): time: " << simulation.world()->getTime() << " \n";

        if (local_robot) {
            local_robot->clear_controllers();
        }

        simulation.clear_robots();
        local_robot.reset();
        simulation.world()->reset();
        simulation.world()->setTime(0);

        local_robot = global2::global_robot->clone();
        local_robot->skeleton()->setPosition(5, 0.15);

        //this controller is dummy but alters state of robot in such a way that it is possible to do manual control
        //without it, hexapod's position will be fixed regardless of legs
        //would be nice to extract this logic and get rid of this dummy conroller
        std::vector<double> ctrl = {1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0.5, 0.5, 0.25, 0.75, 0.5, 1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5, 1, 0.5, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5};
        local_robot->add_controller(std::make_shared<robot_dart::control::HexaControl>(step_duration, ctrl));
        std::static_pointer_cast<robot_dart::control::HexaControl>(local_robot->controllers()[0])
                ->set_h_params(std::vector<double>(1, step_duration));

        simulation.add_robot(local_robot);

        reward_accumulator = 0;
        initial_position = local_robot->skeleton()->getPositions().head(6).tail(3).cast<float>();

//        std::cout << "init pos " << initial_position << std::endl;

        Mat obs{get_obs()};

        old_obs = obs;

        return std::move(obs);
    }

    std::vector<Mat> step(const Mat &actions) override {

        assert(actions.cols() == action_space_size);

        //std::cout << actions << std::endl;

        Eigen::Vector3f pos_before_step {local_robot->skeleton()->getPositions().head(6).tail(3).cast<float>()};

/*            std::cout << index << std::endl;
            for (int i = 0; i<18; i++) {
                std::cout << angles2[i+1]<< (i==17?"":", ");
            }
            std::cout << std::endl;*/

        Eigen::VectorXd target_positions = Eigen::VectorXd::Zero(action_space_size + 6);
        for (size_t i = 0; i < action_space_size; i++)
            target_positions(i + 6) = ((i % 3 == 1) ? 1.0 : -1.0) * Env::clamp(actions(0,i),min_action_value,max_action_value);

        //std::cout << target_positions << std::endl;

        Eigen::VectorXd q = local_robot->skeleton()->getPositions();
        Eigen::VectorXd q_err = target_positions - q;

        double gain = 1.0 / (dart::math::constants<double>::pi() * step_duration);

        Eigen::VectorXd commands = q_err * gain;

        commands.head(6) = Eigen::VectorXd::Zero(6);

        local_robot->skeleton()->setCommands(commands);

        simulation.world()->step(false);

        Eigen::Vector3f pos_after_step {local_robot->skeleton()->getPositions().head(6).tail(3).cast<float>()};

        auto s = (pos_after_step - pos_before_step);

//        std::cout << "step distance by axis:" << s.transpose() << std::endl;
//        std::cout << "total distance:" << s.norm() << std::endl;
//        std::cout << "velocity by axis:" << s/duration.Get() << std::endl;

        Mat rewards {Mat(1,1)};
        rewards(0,0) = s[0];

        reward_accumulator+=rewards(0,0);

        bool done{get_time()>=simulation_duration};

        Mat dones {Mat(1,1)};
        dones(0,0) = done?1.f:0.f;

        Mat obs {get_obs()};

//        std::cout << rewards << std::endl;
//        std::cout << s << std::endl;
//        std::cout << obs << std::endl;


        //this should be really handled outside
        if(done){
            float reward_error = std::abs((pos_after_step - initial_position)(0,0) - reward_accumulator);

            obs = reset();

            if(reward_error>1e-2){
                std::cout << "Error: Episode reward does not agree with sum of rewards by value of: " << reward_error << std::endl;
                assert(false);
            }

        }

        old_rew  = rewards;
        old_obs = obs;

        return {obs,rewards,dones};
    }

    void render() override {
        simulation.graphics()->refresh();
    }

    float get_time() override {
        return simulation.world()->getTime();
    }

    Mat get_original_obs() override {
        return old_obs;
    }

    Mat get_original_rew() override {
        return old_rew;
    }

    void serialize(nlohmann::json& json) override {

    }

    void deserialize(nlohmann::json& json) override {

    }

protected:
    virtual Mat get_obs() {
        Mat obs {Mat(1,1)};
        obs(0,0) = fmod(get_time(),1);
        return std::move(obs);
    }

protected:
    std::shared_ptr<robot_dart::Robot> local_robot;

private:

    static const int action_space_size = 18;
    static const int observation_space_size = 1;

    float step_duration;
    float simulation_duration;
    robot_dart::RobotDARTSimu simulation;
    float min_action_value;
    float max_action_value;
    float reward_accumulator;
    Eigen::Vector3f initial_position;

protected:
    Mat old_obs;
    Mat old_rew;
};

#endif //PPO_CPP_HEXAPOD_ENV_HPP
