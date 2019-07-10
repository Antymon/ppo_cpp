#ifndef __NETWORK__LOADER__HPP__
#define __NETWORK__LOADER__HPP__

//#include <sferes/misc/rand.hpp>

//#include "tensorflow/core/framework/graph.h"
#include "tensorflow/core/framework/tensor.pb.h"
//#include "tensorflow/core/graph/default_device.h"
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

using namespace tensorflow;


class NetworkLoader {

    struct Options {
        // config setting
        static const int input_dim = 100;
        static const int batch_size = 20000;
        static const int nb_epoch = 25000;
        static constexpr float CV_fraction = 0.75;
    };

public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

    NetworkLoader(std::string graph_path) : _global_step(0) {
        // Initialize a tensorflow session
        std::cout << "start initalize session and loading graph" << "\n";
        // First we load and initialize the model.
        Status load_graph_status = load_graph(graph_path);
        if (!load_graph_status.ok()) {
            LOG(ERROR) << load_graph_status;
        }
        std::cout << "loading done" << std::endl;

    }

    int32 _global_step;

    // Reads a model graph definition from disk, and creates a session object you
    // can use to run it.
    Status load_graph(string graph_file_name) {
        MetaGraphDef graph_def;
        SessionOptions opts;
        //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(1);
        //opts.config.mutable_gpu_options()->set_allow_growth(true);
        //(*opts.config.mutable_device_count())["GPU"]=0;//causes segfault...
        //opts.config.set_allow_soft_placement(true);

        _session.reset(NewSession(opts));

        Status status = ReadTextProto(Env::Default(), graph_file_name, &graph_def);
        if (!status.ok()) {
            std::cout << "Error reading graph definition" << graph_file_name << ": " << status.ToString() << std::endl;
        }

        // Add the graph to the session
        status = _session->Create(graph_def.graph_def());
        if (!status.ok()) {
            std::cout << "Error creating graph: " << status.ToString() << std::endl;
        }

        std::vector<Tensor> outputs;
        status = _session->Run({}, {}, {"oh_init"}, &outputs);


        if (!status.ok())
            std::cout << status.ToString() << "\n";
        else
            std::cout << "Success load graph !! " << "\n";

        // Read weights from the saved checkpoint
//        Tensor checkpointPathTensor(DT_STRING, TensorShape());
//        checkpointPathTensor.scalar<std::string>()() = graph_file_name;
//
//        status = _session->Run(
//                {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor},},
//                {},
//                {graph_def.saver_def().restore_op_name()},
//                nullptr);
//        if (!status.ok())
//            std::cout << "Error loading checkpoint from " << graph_file_name << ": " << status.ToString() << std::endl;
//        else
//            std::cout << "Success load weights !! " << "\n";
//
        return Status::OK();
    }

    void reset_optimizer() {
        std::cout << "reset_optimizer" << std::endl;
        std::vector<Tensor> outputs_reset;
        TensorShape shape;
        shape.InsertDim(0, 1);
        //Tensor global_step(DT_INT32,TensorShape({1}));
        Tensor global_step(DT_INT32, shape);
        auto global_step_mapped = global_step.tensor<int32, 1>();
        global_step_mapped(0) = _global_step;

        Status status = _session->Run({{"step_id", global_step}}, {}, {"reset_optimizer"}, &outputs_reset);

        if (!status.ok()) {
            LOG(ERROR) << status;
        }
        std::cout << "reset_optimizer done" << std::endl;
    }

    void prepare_batches(std::vector<std::vector<std::pair<std::string, Tensor> > > &batches, const Mat &data) const {
        if (data.rows() <= Options::batch_size)
            batches = std::vector<std::vector<std::pair<std::string, Tensor> > >(1);
        else
            batches = std::vector<std::vector<std::pair<std::string, Tensor> > >(
                    floor(data.rows() / (Options::batch_size)));

        if (batches.size() == 1)
            create_feed(batches[0], data, 1, 0);
        else
            for (int ind = 0; ind < batches.size(); ind++)
                create_feed(batches[ind], data.middleRows(ind * Options::batch_size, Options::batch_size), 1, 0);
    }

    void create_feed(std::vector<std::pair<std::string, Tensor>> &feed, const Mat &data, float keep, int step) const {
        TensorShape shape;
        shape.InsertDim(0, data.rows());
        shape.InsertDim(1, Options::input_dim);

        Tensor inputs(DT_FLOAT, shape);
        convert_mat(data, inputs);

        TensorShape shape_1;
        shape_1.InsertDim(0, 1);

        Tensor keep_prob(DT_FLOAT, shape_1);

        auto keep_prob_mapped = keep_prob.tensor<float, 1>();
        keep_prob_mapped(0) = keep;

        Tensor global_step(DT_INT32, TensorShape());
        auto global_step_mapped = global_step.tensor<int32, 0>();
        global_step_mapped(0) = step;

        feed = {
                {"input_x",   inputs},
                {"keep_prob", keep_prob},
                {"step_id",   global_step}
        };

    }

    void split_dataset(const Mat &data, Mat &train, Mat &valid) {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.rows());
        perm.setIdentity();
        std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
        Mat tmp = perm * data;

        size_t l_train = data.rows() > 500 ? std::floor(data.rows() * Options::CV_fraction) : data.rows();
        size_t l_valid = data.rows() > 500 ? data.rows() - l_train : data.rows();

        assert(l_train != 0 && l_valid != 0);

        train = tmp.topRows(l_train);
        valid = tmp.bottomRows(l_valid);
        std::cout << "training dataset: " << train.rows() << "  valid dataset: " << valid.rows() << std::endl;
    }


    float training(const Mat &data, bool full_train = false) {

        reset_optimizer();


        Mat train_db, valid_db;
        for (size_t repl = 0; repl < 5; repl++) {
            split_dataset(data, train_db, valid_db);

            float init_tr_recon_loss = get_avg_recon_loss(train_db);
            float init_vl_recon_loss = get_avg_recon_loss(valid_db);
            std::cout << "INIT train loss: " << init_tr_recon_loss << "  INIT validation loss: " << init_vl_recon_loss
                      << std::endl;

            std::vector<std::vector<std::pair<std::string, Tensor> > > batches;
            prepare_batches(batches, train_db);


            bool _continue = true;

            Eigen::VectorXd previous_avg = Eigen::VectorXd::Ones(5) * 100;


            for (int epoch = 0; epoch < Options::nb_epoch && _continue; epoch++) {

                for (int it = 0; it < batches.size(); it++) {

                    std::vector<Tensor> outputs_train;
                    auto global_step_mapped = batches[it].back().second.tensor<int32, 0>();
                    global_step_mapped(0) = _global_step;
                    Status status = _session->Run(batches[it], {}, {"train_step"}, &outputs_train);

                    if (!status.ok()) {
                        LOG(ERROR) << status;
                    }

                }

                _global_step++;
                if (!full_train && epoch % 100 == 0) {
                    float current_avg = get_avg_recon_loss(valid_db);
                    for (size_t t = 1; t < previous_avg.size(); t++)
                        previous_avg[t - 1] = previous_avg[t];
                    previous_avg[previous_avg.size() - 1] = current_avg;
                    if ((previous_avg.array() - previous_avg[0]).mean() > 0 &&
                        get_avg_recon_loss(train_db) < init_tr_recon_loss)
                        _continue = false;
                }
                if (epoch % 1000 == 0)
                    std::cout << epoch << "  valid: " << get_avg_recon_loss(valid_db) << " train: "
                              << get_avg_recon_loss(train_db) << std::endl;


            }
            std::cout << repl << " Final train loss: " << get_avg_recon_loss(train_db) << "  Final validation loss: "
                      << get_avg_recon_loss(valid_db) << std::endl;
        }
        std::cout << "Final loss: " << get_avg_recon_loss(data) << std::endl;
        //return (get_avg_recon_loss(train_db) * Options::CV_fraction + get_avg_recon_loss(valid_db) *(1-Options::CV_fraction));
        return get_avg_recon_loss(data);

    }


    void get_reconstruction(const Mat &data, Mat &res) const {
        Mat desc, recon_loss, loss;
        eval(data, desc, recon_loss, loss, res);
        //oo << "average loss is : " << avg_loss << "   average recon_loss : " << recon_loss.mean() << std::endl;
        //oo<<recon_loss.transpose()<<std::endl<<desc.transpose()<<std::endl<<data.transpose()<<std::endl<<res.transpose()<<std::endl;

    }


    float get_avg_recon_loss(const Mat &data) const {
        Mat desc, recon_loss, loss, reconst;
        eval(data, desc, recon_loss, loss, reconst);
        return recon_loss.mean();
    }


    void eval(const Mat &data, Mat &desc, Mat &recon_loss, Mat &loss, Mat &reconst) const {

        std::vector<std::pair<std::string, Tensor>> eval_data;
        create_feed(eval_data, data, 1, _global_step);

        std::vector<Tensor> outputs;
        Status status = _session->Run({eval_data},
                                      {"encoder/latent", "loss", "recon_loss", "decoder/decoded", "learning_rate"}, {},
                                      &outputs);
        if (!status.ok()) {
            LOG(ERROR) << status;
        }


        convert_tensor(outputs[0], desc);
        convert_tensor(outputs[1], loss);
        convert_tensor(outputs[2], recon_loss);
        convert_tensor(outputs[3], reconst);
        Mat lr;
        convert_tensor(outputs[4], lr);
    }


private:
    std::unique_ptr<Session> _session;

    void convert_tensor(Tensor &t, Mat &m) const {
        assert(t.dims() < 3); //not suported
        switch (t.dims()) {
            case 0:
                m = Mat(1, 1); //scalar
                break;
            case 1:
                m = Mat(t.dim_size(0), 1);
                break;
            case 2:
                m = Mat(t.dim_size(0), t.dim_size(1));
                break;
        }

        auto src = t.flat<float>().data();
        memcpy(m.data(), src, m.cols() * m.rows() * sizeof(float));


    }


    void convert_mat(const Mat &m, Tensor &t) const {
        TensorShape shape;
        shape.InsertDim(0, m.rows());
        shape.InsertDim(1, m.cols());

        t = Tensor(DT_FLOAT, shape);
        auto dst = t.flat<float>().data();
        memcpy(dst, m.data(), m.cols() * m.rows() * sizeof(float));

    }

};


#endif
