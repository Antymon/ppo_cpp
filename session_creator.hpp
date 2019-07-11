#ifndef __SESSION__CREATOR__HPP__
#define __SESSION__CREATOR__HPP__


#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include <tensorflow/core/platform/env.h>
#include "tensorflow/core/public/session.h"
#include <tensorflow/core/protobuf/meta_graph.pb.h>


#include <chrono>

class SessionCreator {

public:

    std::unique_ptr<tensorflow::Session> load_graph(std::string graph_file_name) {

        std::unique_ptr<tensorflow::Session> session;

        // Initialize a tensorflow session
        std::cout << "start initalize session and loading graph" << "\n";
        // First we load and initialize the model.

        tensorflow::MetaGraphDef graph_def;
        tensorflow::SessionOptions opts;
        //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(1);
        //opts.config.mutable_gpu_options()->set_allow_growth(true);
        //(*opts.config.mutable_device_count())["GPU"]=0;//causes segfault...
        opts.config.set_allow_soft_placement(true);


        session.reset(NewSession(opts));

        tensorflow::Status status = ReadTextProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
        if (!status.ok()) {
            std::cout << "Error reading graph definition" << graph_file_name << ": " << status.ToString() << std::endl;
            return nullptr;
        }

        // Add the graph to the session
        status = session->Create(graph_def.graph_def());
        if (!status.ok()) {
            std::cout << "Error creating graph: " << status.ToString() << std::endl;
            return nullptr;
        }

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {}, {"init"}, &outputs);

        if (!status.ok()) {
            std::cout << status.ToString() << "\n";
            return nullptr;
        }
        else
            std::cout << "Success load graph !! " << "\n";

        std::cout << "loading done" << std::endl;

        return session;
    }
};


#endif
