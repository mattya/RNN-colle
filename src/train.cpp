#include "core/network.hpp"
#include "core/io.hpp"
#include "core/util.hpp"
#include "../picojson/picojson.h"

#include <cstdio>
#include <iostream>
#include <string>
using namespace std;

typedef unsigned char byte;



int main(){
  InitTensorEngine<cpu>();

  	picojson::value v;
  	cin >> v;
	if (std::cin.fail()) {
		std::cerr << picojson::get_last_error() << std::endl;
		return 1;
	}

	picojson::object& o = v.get<picojson::object>();
	string network_type;
	picojson::value network_param;
	string mode="train";

// data
	string data_type;
	string fn_train_data;
	string fn_train_label;
	string fn_test_data;
	string fn_test_label;
	string fn_predict;
	int n_train, n_test;

// learning
	int n_time=1, n_batch=1;
	int n_x_data=1, n_x_label=1;
	int init_epoch = 0;
	int iter_per_epoch=100;
	int snapshot_interval=100;
	int max_epoch=10000;
	bool load_model=false;
	string sgd = "rmsprop";
	float momentum = 0.95f;
	float decay = 0.001f;
	float base_lr = 0.0001f;
	float lr_mult = 0.5f;
	int lr_mult_interval = 100;
	int sampling_length = 0;
	int train_error_limit = 0;


	for (picojson::object::const_iterator i = o.begin(); i != o.end(); ++i) {
		if(i->first == "env"){
			picojson::object env = i->second.get<picojson::object>();
			for (picojson::object::const_iterator j = env.begin(); j != env.end(); ++j) {
				if(j->first == "python") Global_params::python = j->second.get<string>();
				if(j->first == "prefix") Global_params::prefix = j->second.get<string>();
				if(j->first == "debug") Global_params::DEBUG = j->second.get<bool>();
				if(j->first == "mode") mode = j->second.get<string>();
			}
		}
		if(i->first == "network"){
			picojson::object net = i->second.get<picojson::object>();
			for (picojson::object::const_iterator j = net.begin(); j != net.end(); ++j) {
				if(j->first == "type") network_type = j->second.get<string>();
				if(j->first == "param") network_param = j->second;
			}
		}
		if(i->first == "data"){
			picojson::object dat = i->second.get<picojson::object>();
			for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
				if(j->first == "type") data_type = j->second.get<string>();
				if(j->first == "train_data") fn_train_data = j->second.get<string>();
				if(j->first == "train_label") fn_train_label = j->second.get<string>();
				if(j->first == "test_data") fn_test_data = j->second.get<string>();
				if(j->first == "test_label") fn_test_label = j->second.get<string>();
				if(j->first == "predict") fn_predict = j->second.get<string>();
				if(j->first == "n_train") n_train = (int)j->second.get<double>();
				if(j->first == "n_test") n_test = (int)j->second.get<double>();
				if(j->first == "n_x_data") n_x_data = (int)j->second.get<double>();
				if(j->first == "n_x_label") n_x_label = (int)j->second.get<double>();
			}
		}
		if(i->first == "learning"){
			picojson::object dat = i->second.get<picojson::object>();
			for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {

				if(j->first == "n_time") n_time = (int)j->second.get<double>();
				if(j->first == "n_batch") n_batch = (int)j->second.get<double>();
				if(j->first == "load_model") load_model = j->second.get<bool>();
				if(j->first == "init_epoch") init_epoch = (int)j->second.get<double>();
				if(j->first == "iter_per_epoch") iter_per_epoch = (int)j->second.get<double>();
				if(j->first == "snapshot_interval") snapshot_interval = (int)j->second.get<double>();
				if(j->first == "max_epoch") max_epoch = (int)j->second.get<double>();
				if(j->first == "sgd") sgd = j->second.get<string>();
				if(j->first == "momentum") momentum = (float)j->second.get<double>();
				if(j->first == "decay") decay = (float)j->second.get<double>();
				if(j->first == "base_lr") base_lr = (float)j->second.get<double>();
				if(j->first == "lr_mult") lr_mult = (float)j->second.get<double>();
				if(j->first == "lr_mult_interval") lr_mult_interval = (int)j->second.get<double>();
				if(j->first == "sampling_length") sampling_length = (int)j->second.get<double>();
				if(j->first == "train_error_limit") train_error_limit = (int)j->second.get<double>();
				if(j->first == "handover") Global_params::handover = j->second.get<bool>();

			}
		}
	}

	cerr << "load json done" << endl;

	// data load
	Node<cpu> *node_train_data, *node_train_label, *node_test_data, *node_test_label;
	Node<cpu> *node_train_data0, *node_test_data0;

	if(data_type=="MNIST"){
		node_train_data = new Node<cpu>(n_train, n_batch, n_time, 784, true);
		node_train_label = new Node<cpu>(n_train, n_batch, n_time, 10, true);
		node_test_data = new Node<cpu>(n_test, n_batch, n_time, 784, true);
		node_test_label = new Node<cpu>(n_test, n_batch, n_time, 10, true);
		read_mnist_data(fn_train_data, node_train_data, n_train, 784, 16);
		read_mnist_label(fn_train_label, node_train_label, n_train, 10, 8);
		read_mnist_data(fn_test_data, node_test_data, n_test, 784, 16);
		read_mnist_label(fn_test_label, node_test_label, n_test, 10, 8);
	}else if(data_type=="series_ascii"){
		node_train_data = new Node<cpu>(n_train, n_batch, n_time, n_x_data, true);
		node_train_label = new Node<cpu>(n_train, n_batch, n_time, n_x_label, true);
		node_test_data = new Node<cpu>(n_test, n_batch, n_time, n_x_data, true);
		node_test_label = new Node<cpu>(n_test, n_batch, n_time, n_x_label, true);
		read_float_data<cpu>(fn_train_data, node_train_data, n_train, n_time, n_x_data);
		read_float_data<cpu>(fn_train_label, node_train_label, n_train, n_time, n_x_label);
		read_float_data<cpu>(fn_test_data, node_test_data, n_test, n_time, n_x_data);
		read_float_data<cpu>(fn_test_label, node_test_label, n_test, n_time, n_x_label);
	}else if(data_type=="series_binary"){
		node_train_data = new Node<cpu>(n_train, n_batch, n_time, n_x_data, true);
		node_train_label = new Node<cpu>(n_train, n_batch, n_time, n_x_label, true);
		node_test_data = new Node<cpu>(n_test, n_batch, n_time, n_x_data, true);
		node_test_label = new Node<cpu>(n_test, n_batch, n_time, n_x_label, true);
		read_float_binary_data<cpu>(fn_train_data, node_train_data, n_train, n_time, n_x_data);
		read_float_binary_data<cpu>(fn_train_label, node_train_label, n_train, n_time, n_x_label);
		read_float_binary_data<cpu>(fn_test_data, node_test_data, n_test, n_time, n_x_data);
		read_float_binary_data<cpu>(fn_test_label, node_test_label, n_test, n_time, n_x_label);
	}else if(data_type=="series_int_binary"){
		node_train_data = new Node<cpu>(n_train, n_batch, n_time, n_x_data, true);
		node_train_label = new Node<cpu>(n_train, n_batch, n_time, n_x_label, true);
		node_test_data = new Node<cpu>(n_test, n_batch, n_time, n_x_data, true);
		node_test_label = new Node<cpu>(n_test, n_batch, n_time, n_x_label, true);
		read_int_binary_data<cpu>(fn_train_data, node_train_data, n_train, n_time, n_x_data);
		read_int_binary_data<cpu>(fn_train_label, node_train_label, n_train, n_time, n_x_label);
		read_int_binary_data<cpu>(fn_test_data, node_test_data, n_test, n_time, n_x_data);
		read_int_binary_data<cpu>(fn_test_label, node_test_label, n_test, n_time, n_x_label);
	}


	// prepare network
	Network<cpu> *network = new Network<cpu>();
	network->train_data = node_train_data;
	network->train_label = node_train_label;
	network->test_data = node_test_data;
	network->test_label = node_test_label;

	int layers = 3;
	int layers2=0;
	int base_neurons=512;
	int input_dim = 1;
	int output_dim = 1;
	string out_nl="none";
	string loss = "mse";
	string shuffle = "full_shuffle";
	int *num_neurons;
	string hidden_nl="tanh";
	bool dropout=false;
	float scale=0.9;

	picojson::object dat = network_param.get<picojson::object>();
	for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
		if(j->first == "layers") layers = (int)j->second.get<double>();
		if(j->first == "layers2") layers2 = (int)j->second.get<double>();
		if(j->first == "neurons"){
			num_neurons = (int *)malloc(layers*sizeof(int));
			const picojson::array& a = j->second.get<picojson::array>();
			int cnt = 0;
			for(picojson::array::const_iterator i=a.begin(); i!=a.end(); ++i){
				num_neurons[cnt++] = (int)i->get<double>();
			}
		}
		if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
		if(j->first == "hidden_nl") hidden_nl = j->second.get<string>();
		if(j->first == "out_nl") out_nl = j->second.get<string>();
		if(j->first == "dropout") dropout = j->second.get<bool>();
		if(j->first == "loss") loss = j->second.get<string>();
		if(j->first == "shuffle") shuffle = j->second.get<string>();
		if(j->first == "scale")scale = (float)j->second.get<double>();
	}
	network->loss_type = loss;
	network->shuffle_type = shuffle;

	if(network_type=="MLP"){
		network->net = new MLP<cpu>(layers, num_neurons, out_nl, hidden_nl);
	}else if(network_type=="Reservoir"){
		network->net = new Reservoir<cpu>(n_x_data, base_neurons, n_x_label, scale);
	}else if(network_type=="Reservoir_MLP"){
		network->net = new Reservoir_MLP<cpu>(n_x_data, base_neurons, n_x_label, layers, num_neurons, scale);
	}else if(network_type=="Stacked_LSTM"){
		network->net = new Stacked_LSTM<cpu>(n_x_data, n_x_label, base_neurons, layers);
	}else if(network_type=="Stacked_LSTM_MLP"){
		network->net = new Stacked_LSTM_MLP<cpu>(n_x_data, base_neurons, num_neurons[0], n_x_label, layers, layers2, num_neurons, out_nl);
	}

	/*else if(network_type=="SLSTM"){
		int layers = 0;
		int base_neurons=512;
		int inout_dim = 1001;
		string out_nl="none";
		string loss = "mse";
		string shuffle = "w2v";

		picojson::object dat = network_param.get<picojson::object>();
		for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
			if(j->first == "layers") layers = (int)j->second.get<double>();
			if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
			if(j->first == "out_nl") out_nl = j->second.get<string>();
			if(j->first == "loss") loss = j->second.get<string>();
			if(j->first == "shuffle") shuffle = j->second.get<string>();
			if(j->first == "inout_dim") inout_dim = (int)j->second.get<double>();
		}
		network->loss_type = loss;
		network->shuffle_type = shuffle;
//		network->is_dropout = dropout;
		cerr << base_neurons << " " << layers << endl;
		network->net = new Stacked_LSTM<cpu>(inout_dim, base_neurons, layers, out_nl);
	}else if(network_type=="HRes"){
		int base_neurons=512;
		int inout_dim = 1001;
		string out_nl="none";
		string loss = "mse";
		string shuffle = "w2v";

		picojson::object dat = network_param.get<picojson::object>();
		for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
			if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
			if(j->first == "out_nl") out_nl = j->second.get<string>();
			if(j->first == "loss") loss = j->second.get<string>();
			if(j->first == "shuffle") shuffle = j->second.get<string>();
			if(j->first == "inout_dim") inout_dim = (int)j->second.get<double>();
		}
		network->loss_type = loss;
		network->shuffle_type = shuffle;
//		network->is_dropout = dropout;
//		cerr << base_neurons << " " << layers << endl;
		network->net = new Hybrid_LSTM_Reservoir<cpu>(inout_dim, base_neurons, out_nl);
	}else if(network_type=="GMR2"){
		int base_neurons=512;
		int inout_dim = 1001;
		string out_nl="none";
		string loss = "mse";
		string shuffle = "w2v";

		picojson::object dat = network_param.get<picojson::object>();
		for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
			if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
			if(j->first == "out_nl") out_nl = j->second.get<string>();
			if(j->first == "loss") loss = j->second.get<string>();
			if(j->first == "shuffle") shuffle = j->second.get<string>();
			if(j->first == "inout_dim") inout_dim = (int)j->second.get<double>();
		}
		network->loss_type = loss;
		network->shuffle_type = shuffle;
//		network->is_dropout = dropout;
//		cerr << base_neurons << " " << layers << endl;
		network->net = new Gate_MLP_Reservoir2<cpu>(inout_dim, base_neurons, out_nl);
	}else if(network_type=="GMR2S"){
		int base_neurons=512;
		int inout_dim = 1001;
		string out_nl="none";
		string loss = "mse";
		string shuffle = "w2v";

		picojson::object dat = network_param.get<picojson::object>();
		for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
			if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
			if(j->first == "out_nl") out_nl = j->second.get<string>();
			if(j->first == "loss") loss = j->second.get<string>();
			if(j->first == "shuffle") shuffle = j->second.get<string>();
			if(j->first == "inout_dim") inout_dim = (int)j->second.get<double>();
		}
		network->loss_type = loss;
		network->shuffle_type = shuffle;
//		network->is_dropout = dropout;
//		cerr << base_neurons << " " << layers << endl;
		network->net = new Gate_MLP_Reservoir2_SLSTM<cpu>(inout_dim, base_neurons, out_nl);
	}else if(network_type=="Hatsuwa_simple"){
		int base_neurons=512;
		int inout_dim = 1001;
		string out_nl="none";
		string loss = "mse";
		string shuffle = "w2v";

		picojson::object dat = network_param.get<picojson::object>();
		for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
			if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
			if(j->first == "out_nl") out_nl = j->second.get<string>();
			if(j->first == "loss") loss = j->second.get<string>();
			if(j->first == "shuffle") shuffle = j->second.get<string>();
			if(j->first == "inout_dim") inout_dim = (int)j->second.get<double>();
		}
		network->loss_type = loss;
		network->shuffle_type = shuffle;
//		network->is_dropout = dropout;
//		cerr << base_neurons << " " << layers << endl;
		network->net = new Hatsuwa_simple<cpu>();
	}else if(network_type=="Hatsuwa_ngram"){
		int base_neurons=512;
		int inout_dim = 1001;
		string out_nl="none";
		string loss = "mse";
		string shuffle = "w2v";

		picojson::object dat = network_param.get<picojson::object>();
		for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
			if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
			if(j->first == "out_nl") out_nl = j->second.get<string>();
			if(j->first == "loss") loss = j->second.get<string>();
			if(j->first == "shuffle") shuffle = j->second.get<string>();
			if(j->first == "inout_dim") inout_dim = (int)j->second.get<double>();
		}
		network->loss_type = loss;
		network->shuffle_type = shuffle;
//		network->is_dropout = dropout;
//		cerr << base_neurons << " " << layers << endl;
		network->net = new Hatsuwa_ngram<cpu>();
	}else if(network_type=="Hatsuwa_aws_res"){
		int base_neurons=512;
		int inout_dim = 1001;
		int layers = 0;
		string out_nl="none";
		string loss = "mse";
		string shuffle = "w2v";

		picojson::object dat = network_param.get<picojson::object>();
		for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
			if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
			if(j->first == "out_nl") out_nl = j->second.get<string>();
			if(j->first == "loss") loss = j->second.get<string>();
			if(j->first == "shuffle") shuffle = j->second.get<string>();
			if(j->first == "inout_dim") inout_dim = (int)j->second.get<double>();
			if(j->first == "layers") layers = (int)j->second.get<double>();
		}
		network->loss_type = loss;
		network->shuffle_type = shuffle;
//		network->is_dropout = dropout;
//		cerr << base_neurons << " " << layers << endl;
		network->net = new Hatsuwa_aws_res<cpu>(base_neurons, base_neurons, layers);
	}else if(network_type=="Hatsuwa_aws_nores"){
		int base_neurons=512;
		int inout_dim = 1001;
		int layers = 0;
		string out_nl="none";
		string loss = "mse";
		string shuffle = "w2v";

		picojson::object dat = network_param.get<picojson::object>();
		for (picojson::object::const_iterator j = dat.begin(); j != dat.end(); ++j) {
			if(j->first == "base_neurons")base_neurons = (int)j->second.get<double>();
			if(j->first == "out_nl") out_nl = j->second.get<string>();
			if(j->first == "loss") loss = j->second.get<string>();
			if(j->first == "shuffle") shuffle = j->second.get<string>();
			if(j->first == "inout_dim") inout_dim = (int)j->second.get<double>();
			if(j->first == "layers") layers = (int)j->second.get<double>();
		}
		network->loss_type = loss;
		network->shuffle_type = shuffle;
//		network->is_dropout = dropout;
//		cerr << base_neurons << " " << layers << endl;
		network->net = new Hatsuwa_aws_nores<cpu>(base_neurons, base_neurons, layers);
	}

	*/
	network->net->set_param("eta", base_lr);
	network->net->set_param("decay", decay);
	network->net->set_param("sgd_algo", sgd=="momentum"?0:1);
	network->net->set_param("momentum", momentum);

	cerr << "network initialize done" << endl;

	// learning

	if(load_model){
		FILE *ii = fopen(to_string("./tmp/"+Global_params::prefix+"_model", init_epoch).c_str(), "rb");
		network->load_model(ii);
		fclose(ii);
	}

	if(mode=="train"){

		for(int epoch=init_epoch+1; epoch<=max_epoch; epoch++){
			network->train(1, iter_per_epoch);

			float train_err = network->train_error(train_error_limit);
			float test_err = network->test_error();
			cerr << "epoch: " << epoch << ", train error: " << train_err << ", test error: " << test_err << endl;
			printf("epoch: %d, train error: %f, test error: %f\n", epoch, train_err, test_err);
			if(epoch%lr_mult_interval==0){
				float eta = network->net->get_param("eta");
		        network->net->set_param("eta", eta*lr_mult);
				cerr << "eta: " << eta << endl;
			}
			if(epoch%snapshot_interval==0){
				FILE *ir = fopen(to_string("./tmp/"+Global_params::prefix+"_model", epoch).c_str(), "wb");
				network->save_model(ir);
				fclose(ir);
			}

		}

	}else if(mode=="predict"){
		network->predict(fn_predict, n_test);
	}

  ShutdownTensorEngine<cpu>();
	return 0;
}
