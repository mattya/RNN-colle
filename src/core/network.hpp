#ifndef NETWORK_H
#define NETWORK_H

#include "mshadow/tensor.h"
#include <iostream>
#include <vector>
using namespace std;
#include "util.hpp"
#include "node.hpp"
#include "layer.hpp"
#include "io.hpp"


using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// TBA 時系列データが複数ある時（先頭フラグがある）
template<typename xpu>
class Network{
public:
	Node<xpu> *train_data, *train_label, *test_data, *test_label;
	Layer<xpu> *net;

	string loss_type;  // mse, category, nll
	string shuffle_type;  // full_shuffle (usually N_time==1), single_series, multiple_series

	Network(string lt="mse", string st="full_shuffle"){
//		cerr << lt << " " << st << endl;
		loss_type = lt;
		shuffle_type = st;
	}

	~Network(){
//		delete out_node;
	}

	void save_model(FILE *fo){
		net->save_model(fo);
	}
	void load_model(FILE *fo){
		net->load_model(fo);
	}

	void train(int epoch_max, int iter_per_epoch){
		int N_train = train_data->N_all;
		int N_batch = train_data->N_batch;
		int N_time = train_data->N_time;
		int N_label = train_label->N_x;

		Global_params::phase_train = true;

		for(int epoch=0; epoch<epoch_max; epoch++){
			Node<xpu> *out_node = new Node<xpu>(N_train, N_batch, N_time, N_label, false);
			net->set_node(train_data, out_node);

			float loss = 0;
			int lst[N_batch];
			if(shuffle_type=="single_series"){
				for(int i=0; i<N_batch; i++) lst[i] = rand()%(N_train-N_time);
			}
			net->set_zero();
			for(int iter=0; iter<iter_per_epoch; iter++){
				if(Global_params::DEBUG) cerr << epoch << " " << iter << endl;
				net->set_zero();

				if(shuffle_type=="full_shuffle"){
					for(int i=0; i<N_batch; i++) lst[i] = rand()%(N_train);
				}else if(shuffle_type=="single_series"){
					for(int i=0; i<N_batch; i++) lst[i] = (lst[i]+N_time)%(N_train-N_time);
				}else if(shuffle_type=="multiple_series"){
					for(int i=0; i<N_batch; i++) lst[i] = (rand()%(N_train/N_time))*N_time;
				}else{
					cerr << "invalid shuffle type" << endl;
				}

				train_data->make_batch(lst);
				train_label->make_batch(lst);

				for(int t=0; t<N_time; t++) net->forward(t);


				if(loss_type=="mse") calc_mse_loss<xpu>(out_node, train_label);
				else if(loss_type=="category") calc_category_loss<xpu>(out_node, train_label);
				else if(loss_type=="nll") calc_xe_loss<xpu>(out_node, train_label);
				else cerr << "invalid loss type" << endl;

				for(int t=N_time-1; t>=0; t--) net->backward(t);

/*
				if(rand()%1==0){
					cout << "iter: " << iter << endl;
					for(int t = 0; t<N_time; t++){
						cout << "t=" << t << endl;
					for(int j=0; j<net->hidden_nodes.size(); j++){
						cout << "----layer: " << j << " ----" << endl;
						Copy(net->hidden_nodes[j]->x_batch_cpu,net->hidden_nodes[j]->x_batch_gpu);
						for(int k=0; k<net->hidden_nodes[j]->N_x; k++){
							printf("%f ", net->hidden_nodes[j]->x_batch_cpu[t][0][k]);
						}
						printf("\n");
					}
					cout << "----layer_out: " << 0 << " ----" << endl;
					Copy(net->out_nodes[0]->x_batch_cpu,net->out_nodes[0]->x_batch_gpu);
					for(int k=0; k<net->out_nodes[0]->N_x; k++){
						printf("%f ", net->out_nodes[0]->x_batch_cpu[t][0][k]);
					}
						printf("\n");

					for(int j=0; j<net->hidden_nodes.size(); j++){
						cout << "----dlayer: " << j << " ----" << endl;
						Copy(net->hidden_nodes[j]->dx_batch_cpu,net->hidden_nodes[j]->dx_batch_gpu);
						for(int k=0; k<net->hidden_nodes[j]->N_x; k++){
							printf("%f ", net->hidden_nodes[j]->dx_batch_cpu[t][0][k]);
						}
						printf("\n");
					}
					cout << "----layer_out: " << 0 << " ----" << endl;
					Copy(net->out_nodes[0]->dx_batch_cpu,net->out_nodes[0]->dx_batch_gpu);
					for(int k=0; k<net->out_nodes[0]->N_x; k++){
						printf("%f ", net->out_nodes[0]->dx_batch_cpu[t][0][k]);
					}
						printf("\n");
					}
				}
*/
				net->update();
			}
			delete out_node;
		}

	}

	// (N_all must be divided by N_batch)
	float train_error(int num=0){
		int N_train = train_data->N_all;
		if(num>0) N_train = num;
		int N_batch = train_data->N_batch;
		int N_time = train_data->N_time;
		int N_label = train_label->N_x;
		Global_params::phase_train = false;
		Node<xpu> *out_node = new Node<xpu>(N_train, N_batch, N_time, N_label, false);
		net->set_node(train_data, out_node);

		float loss = 0;
		int offset = rand()%(train_data->N_all-N_train+1);
		if(shuffle_type=="multiple_series"){
			offset = (rand()%((train_data->N_all-N_train)/N_time))*N_time;
		}
		for(int iter=0; iter<N_train/N_batch/N_time; iter++){
			net->set_zero();
			int lst[N_batch];
			if(shuffle_type=="full_shuffle"){
				for(int i=0; i<N_batch; i++) lst[i] = rand()%(N_train);
			}else{
				for(int i=0; i<N_batch; i++) lst[i] = i*(N_train/N_batch)+iter*N_time+offset;
			}
			train_data->make_batch(lst);
			train_label->make_batch(lst);
			for(int t=0; t<N_time; t++) net->forward(t);
			if(loss_type=="mse") loss += calc_mse_error<xpu>(out_node, train_label);
			else if(loss_type=="category") loss += calc_accuracy<xpu>(out_node, train_label);
			else if(loss_type=="nll") loss += calc_log_likelihood<xpu>(out_node, train_label);
			else cerr << "invalid loss type" << endl;
		}
		delete out_node;
		return loss/(N_train/N_batch/N_time);
	}

	float test_error(){
		int N_test = test_data->N_all;
		int N_batch = test_data->N_batch;
		int N_time = train_data->N_time;
		int N_label = test_label->N_x;

		Global_params::phase_train = false;
//		cerr << N_test << " " << N_batch << " " << N_time << " " << N_label << endl;
		Node<xpu> *out_node = new Node<xpu>(N_test, N_batch, N_time, N_label, false);
		net->set_node(test_data, out_node);

		float loss = 0;
		for(int iter=0; iter<N_test/N_batch/N_time; iter++){
			net->set_zero();
			int lst[N_batch];
			for(int i=0; i<N_batch; i++) lst[i] = i*(N_test/N_batch)+iter*N_time;
			if(shuffle_type=="full_shuffle"){
				for(int i=0; i<N_batch; i++)lst[i] = rand()%(N_test);
			}
				// batch, timeあわせて全データを走査
//			for(int i=0; i<N_batch; i++) lst[i] = iter*(N_batch)+i;
			test_data->make_batch(lst);
			test_label->make_batch(lst);
//			net->forward_batch();
			for(int t=0; t<N_time; t++) net->forward(t);
//				cerr << loss_type << endl;
			if(loss_type=="mse") loss += calc_mse_error<xpu>(out_node, test_label);
			else if(loss_type=="category") loss += calc_accuracy<xpu>(out_node, test_label);
			else if(loss_type=="nll") loss += calc_log_likelihood<xpu>(out_node, test_label);
			else cerr << "invalid loss type" << endl;
			//if(iter%100==99) cerr << iter << ": " << loss << endl;



		}
		delete out_node;
		return loss/(N_test/N_batch/N_time);
	}

	void predict(string filename, int num_step){
		int N_pred = num_step;
		int N_batch = test_data->N_batch;
		int N_time = train_data->N_time;
		int N_label = test_label->N_x;
		Global_params::phase_train = false;
		Node<xpu> *pred_node = new Node<xpu>(N_pred, N_batch, N_time, N_label, true);
		net->set_node(test_data, pred_node);

		// different batch different data
		for(int iter=0; iter<N_pred/N_batch/N_time; iter++){
			net->set_zero();
			int lst[N_batch];
			for(int i=0; i<N_batch; i++) lst[i] = i*(N_pred/N_batch)+iter*N_time;
			test_data->make_batch(lst);
			test_label->make_batch(lst);
			for(int t=0; t<N_time; t++){
				net->forward(t);
				for(int i=0; i<N_batch; i++){
					Copy(pred_node->x_all[0][lst[i]+t], pred_node->x_batch_gpu[t][i]);
				}
			}
		}

		write_float_data<xpu>(filename, pred_node, N_pred, N_label);
		delete pred_node;
	}


};

#endif
