//
// Created by szymon on 13/07/19.
//

#ifndef PPO_CPP_TENSORBOARD_HPP
#define PPO_CPP_TENSORBOARD_HPP

#include <tensorflow/core/util/events_writer.h>
#include <string>
#include <iostream>
#include <tensorflow/core/framework/summary.pb.h>

class TensorboardWriter{
public:
    TensorboardWriter(const std::string& tensorboard_log_path,const std::string& tb_log_name,bool new_tb_log=true)
    : save_path{tensorboard_log_path+tb_log_name}
    , writer{save_path} {

        std::cout << "tb" << std::endl;
        std::cout << save_path << std::endl;

//        for (int i = 0; i < 150; ++i)
//            write_scalar(i * 20, i, "test_scalar", 150.f / i);
    }

    void write_summary(tensorflow::int64 step,const std::string& encoded_summary){
        tensorflow::Event event;
        event.set_wall_time(time(nullptr));
        event.set_step(step);
        event.mutable_summary()->ParseFromString(encoded_summary);

        writer.WriteEvent(event);
    }

    void write_scalar(double wall_time, tensorflow::int64 step, const std::string& tag, float simple_value) {
        tensorflow::Event event;
        event.set_wall_time(wall_time);
        event.set_step(step);
        tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
        summ_val->set_tag(tag);
        summ_val->set_simple_value(simple_value);
        writer.WriteEvent(event);
    }

    void write_scalar(tensorflow::int64 step, const std::string& tag, float simple_value) {
        write_scalar(time(nullptr),step,tag,simple_value);
    }

private:

    std::string save_path;
    tensorflow::EventsWriter writer;

};

#endif //PPO_CPP_TENSORBOARD_HPP
