#ifndef COMP_LAYER_H
#define COMP_LAYER_H

#include "mshadow/tensor.h"
#include <iostream>
#include <vector>
using namespace std;
#include "util.hpp"
#include "layer.hpp"
#include "node.hpp"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

/*
template<typename xpu>
class Hatsuwa_input : public Layer<xpu>{
public:
	int nodes_in=5000, nodes_base=50, nodes_res=500, nodes_out=500;
	float scale0 = 0.9;
	float scale1 = 0.97;
	Hatsuwa_input(){
		int mlp0[4] = {nodes_res, nodes_res, nodes_out};

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_base)));
		this->layers.push_back((new NonLinearLayer<xpu>("tanh")));
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_base)));
		this->layers.push_back((new NonLinearLayer<xpu>("tanh")));

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_base, nodes_res)));
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_res, nodes_res, scale0)));

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_res, nodes_base)));
		this->layers.push_back((new NonLinearLayer<xpu>("tanh")));

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_base, nodes_res)));
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_res, nodes_res, scale1)));

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_base, nodes_out)));
		this->layers.push_back((new MLP<xpu>(3, mlp0, "none")));
		this->layers.push_back((new MLP<xpu>(3, mlp0, "none")));

		this->layers.push_back(new TimeShiftLayer<xpu>());
		this->layers.push_back(new TimeShiftLayer<xpu>());


		this->in_nodes.push_back(NULL);
		this->hidden_nodes.push_back(NULL);  // in_0        base 0
		this->hidden_nodes.push_back(NULL);  // in_1        base 1
		this->hidden_nodes.push_back(NULL);  // res_1       res  2
		this->hidden_nodes.push_back(NULL);  // res_1 (t+1) res  3
		this->hidden_nodes.push_back(NULL);  // in_2        base 4
		this->hidden_nodes.push_back(NULL);  // res_2       res  5
		this->hidden_nodes.push_back(NULL);  // res_2 (t+1) res  6
		this->out_nodes.push_back(NULL);

	}
	// len(this->hidden_nodes) == num_layers-2
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_base, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_base, false);
		this->hidden_nodes[2] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->hidden_nodes[3] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->hidden_nodes[4] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_base, false);
		this->hidden_nodes[5] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->hidden_nodes[6] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[0], this->hidden_nodes[0]);
		this->layers[2]->set_node(this->in_nodes[0], this->hidden_nodes[1]);
		this->layers[3]->set_node(this->hidden_nodes[1], this->hidden_nodes[1]);

		this->layers[4]->set_node(this->hidden_nodes[1], this->hidden_nodes[2]);
		this->layers[5]->set_node(this->hidden_nodes[3], this->hidden_nodes[2]);

		this->layers[6]->set_node(this->hidden_nodes[2], this->hidden_nodes[4]);
		this->layers[7]->set_node(this->hidden_nodes[4], this->hidden_nodes[4]);

		this->layers[8]->set_node(this->hidden_nodes[4], this->hidden_nodes[5]);
		this->layers[9]->set_node(this->hidden_nodes[6], this->hidden_nodes[5]);

		this->layers[10]->set_node(this->hidden_nodes[0], this->out_nodes[0]);
		this->layers[11]->set_node(this->hidden_nodes[2], this->out_nodes[0]);
		this->layers[12]->set_node(this->hidden_nodes[5], this->out_nodes[0]);

		this->layers[13]->set_node(this->hidden_nodes[2], this->hidden_nodes[3]);
		this->layers[14]->set_node(this->hidden_nodes[5], this->hidden_nodes[6]);

		this->hidden_nodes[3]->set_zero();
		this->hidden_nodes[3]->set_zero_mode = 1;
		this->hidden_nodes[6]->set_zero();
		this->hidden_nodes[6]->set_zero_mode = 1;
	}
	void update(){
		for(int i=0; i<this->layers.size(); i++){
			if(i==5 || i==9) continue;
			this->layers[i]->update();
		}
	}
};
*/
template<typename xpu>
class Hatsuwa_input : public Layer<xpu>{
public:
	int nodes_in, nodes_base, nodes_res, nodes_out;
	float scale0;
	float scale1;
	Hatsuwa_input():nodes_in(5000),nodes_base(50),nodes_res(500),scale0(0.9),scale1(0.97){
		int mlp0[4] = {nodes_res, nodes_res, nodes_out};

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_base)));
		this->layers.push_back((new NonLinearLayer<xpu>("tanh")));
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_base)));

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_base, nodes_res)));
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_res, nodes_res, scale0)));

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_base, nodes_out)));
		this->layers.push_back((new MLP<xpu>(3, mlp0, "none")));

		this->layers.push_back(new TimeShiftLayer<xpu>());


		this->in_nodes.push_back(NULL);
		this->hidden_nodes.push_back(NULL);  // in_0        base 0
		this->hidden_nodes.push_back(NULL);  // in_1        base 1
		this->hidden_nodes.push_back(NULL);  // res_1       res  2
		this->hidden_nodes.push_back(NULL);  // res_1 (t+1) res  3
		this->out_nodes.push_back(NULL);

	}
	// len(this->hidden_nodes) == num_layers-2
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_base, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_base, false);
		this->hidden_nodes[2] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->hidden_nodes[3] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[0], this->hidden_nodes[0]);
		this->layers[2]->set_node(this->in_nodes[0], this->hidden_nodes[1]);

		this->layers[3]->set_node(this->hidden_nodes[1], this->hidden_nodes[2]);
		this->layers[4]->set_node(this->hidden_nodes[3], this->hidden_nodes[2]);

		this->layers[5]->set_node(this->hidden_nodes[0], this->out_nodes[0]);
		this->layers[6]->set_node(this->hidden_nodes[2], this->out_nodes[0]);

		this->layers[7]->set_node(this->hidden_nodes[2], this->hidden_nodes[3]);

		this->hidden_nodes[3]->set_zero();
		this->hidden_nodes[3]->set_zero_mode = 1;
	}
	void update(){
		for(int i=0; i<this->layers.size(); i++){
			if(i==4) continue;
			this->layers[i]->update();
		}
	}
};

template<typename xpu>
class Hatsuwa_simple : public Layer<xpu>{
public:
	int inout_neurons, base_neurons, res_neurons;
	Hatsuwa_simple():inout_neurons(16),base_neurons(500),res_neurons(500){
		int mlp0[3] = {500, 500, 5000};

		this->layers.push_back(new FullConnectLayer<xpu>(inout_neurons, 200));
		this->layers.push_back(new NonLinearLayer<xpu>("tanh"));
		this->layers.push_back(new FullConnectLayer<xpu>(200, 500));

		this->layers.push_back(new Gate_MLP_Reservoir<xpu>(inout_neurons,inout_neurons,50,50,res_neurons,500,500, 0.9f));
		this->layers.push_back(new FullConnectLayer<xpu>(500, 500));

		this->layers.push_back(new Gate_MLP_Reservoir<xpu>(res_neurons,res_neurons,50,50,res_neurons,500,500, 0.97f));

		this->layers.push_back(new NonLinearLayer<xpu>("tanh"));

		this->layers.push_back(new Stacked_LSTM2<xpu>(500, 5000, 500, 3, "none"));
//		this->layers.push_back(new MLP<xpu>(3, mlp0, "none"));

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
		// 0 short-circuit
		// 1 first reservoir
		// 2 common hidden
		for(int i=0; i<3; i++){
			this->hidden_nodes.push_back(NULL);
		}
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;

		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 200, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 500, false);
		this->hidden_nodes[2] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 500, false);

		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[0], this->hidden_nodes[0]);
		this->layers[2]->set_node(this->hidden_nodes[0], this->hidden_nodes[2]);
		this->layers[3]->set_node(this->in_nodes[0], this->in_nodes[0], this->hidden_nodes[1]);
		this->layers[4]->set_node(this->hidden_nodes[1], this->hidden_nodes[2]);
		this->layers[5]->set_node(this->hidden_nodes[1], this->hidden_nodes[1], this->hidden_nodes[2]);
		this->layers[6]->set_node(this->hidden_nodes[2], this->hidden_nodes[2]);
		this->layers[7]->set_node(this->hidden_nodes[2], this->out_nodes[0]);
	}

/*
	void save_model(FILE *tmp){
		FILE *ir = fopen(to_string("./tmp/"+Global_params::prefix+"_model_0", 0).c_str(), "wb");
		this->layers[0]->save_model(ir);
		this->layers[1]->save_model(ir);
		this->layers[2]->save_model(ir);
		this->layers[3]->save_model(ir);
		this->layers[4]->save_model(ir);
		this->layers[5]->save_model(ir);
		this->layers[6]->save_model(ir);
		fclose(ir);

		ir = fopen(to_string("./tmp/"+Global_params::prefix+"_model_1", 0).c_str(), "wb");
		this->layers[7]->save_model(ir);
		fclose(ir);
	}
	void load_model(FILE *tmp){
		FILE *ir = fopen(to_string("./tmp/"+Global_params::prefix+"_model_0", 0).c_str(), "rb");
		this->layers[0]->load_model(ir);
		this->layers[1]->load_model(ir);
		this->layers[2]->load_model(ir);
		this->layers[3]->load_model(ir);
		this->layers[4]->load_model(ir);
		this->layers[5]->load_model(ir);
		this->layers[6]->load_model(ir);
		fclose(ir);

		ir = fopen(to_string("./tmp/"+Global_params::prefix+"_model_1", 0).c_str(), "rb");
		this->layers[7]->load_model(ir);
		fclose(ir);
	}
*/
};

template<typename xpu>
class Hatsuwa_aws_res : public Layer<xpu>{
public:
	int lstm_layers;
	int in_neurons, base_neurons, res_neurons, out_neurons;
	Hatsuwa_aws_res(int n_base, int n_res, int n_layers):in_neurons(16),out_neurons(5000),base_neurons(n_base),res_neurons(n_res),lstm_layers(n_layers){
		int mlp0[3] = {n_res,n_base,n_base};
		int mlp1[3] = {1000,1000,out_neurons};

		this->layers.push_back(new Reservoir_MLP<xpu>(in_neurons, res_neurons, 3, mlp0, 0.85, "tanh"));
		this->layers.push_back(new Stacked_LSTM3<xpu>(base_neurons, 1000, base_neurons, lstm_layers));
		this->layers.push_back(new MLP<xpu>(3, mlp1, "none"));

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);

		for(int i=0; i<2; i++){
			this->hidden_nodes.push_back(NULL);
		}
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;

		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, base_neurons, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 1000, false);

		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[0], this->hidden_nodes[1]);
		this->layers[2]->set_node(this->hidden_nodes[1], this->out_nodes[0]);
	}
	/*
	void load_model(FILE *tmp){
		this->layers[0]->load_model(tmp);
		for(int i=0; i<12; i++){
			this->layers[1]->layers[i]->load_model(tmp);
		}
		this->layers[1]->layers[18]->load_model(tmp);

		this->layers[2]->load_model(tmp);
	}
	*/
};

template<typename xpu>
class Hatsuwa_aws_nores : public Layer<xpu>{
public:
	int lstm_layers;
	int in_neurons, base_neurons, res_neurons, out_neurons;
	Hatsuwa_aws_nores(int n_base, int n_res, int n_layers):in_neurons(16),out_neurons(5000),base_neurons(n_base),res_neurons(n_res),lstm_layers(n_layers){
		int mlp0[3] = {in_neurons,n_base,n_base};
		int mlp1[3] = {1000,1000,out_neurons};

		this->layers.push_back(new MLP<xpu>(3, mlp0, "tanh"));
		this->layers.push_back(new Stacked_LSTM3<xpu>(base_neurons, 1000, base_neurons, lstm_layers));
		this->layers.push_back(new MLP<xpu>(3, mlp1, "none"));

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);

		for(int i=0; i<2; i++){
			this->hidden_nodes.push_back(NULL);
		}
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;

		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, base_neurons, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 1000, false);

		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[0], this->hidden_nodes[1]);
		this->layers[2]->set_node(this->hidden_nodes[1], this->out_nodes[0]);
	}
};

/*
template<typename xpu>
class Hatsuwa_simple : public Layer<xpu>{
public:
	int inout_neurons=13, base_neurons=500, res_neurons=500;
	Hatsuwa_simple(){
		int mlp0[4] = {1024, 512, 512, 5000};

		this->layers.push_back(new Reservoir_MLP<xpu>(16, 1024, 4, mlp0, 0.9, "none"));

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
		this->layers[0]->set_node(this->in_nodes[0], this->out_nodes[0]);
	}

};
*/

template<typename xpu>
class Hatsuwa_ngram_only : public Layer<xpu>{
public:
	Hatsuwa_ngram_only(){
		this->layers.push_back(new NonLinearLayer<xpu>("none"));
		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
		this->layers[0]->set_node(this->in_nodes[0], this->out_nodes[0]);
	}
};

template<typename xpu>
class Hatsuwa_ngram : public Layer<xpu>{
public:
	int lstm_layers;
	int in_neurons, base_neurons, res_neurons, out_neurons;
	Hatsuwa_ngram(int n_base, int n_res, int n_layers):in_neurons(16),out_neurons(5000),base_neurons(n_base),res_neurons(n_res),lstm_layers(n_layers){
		int mlp0[3] = {n_res,n_base,n_base};

		this->layers.push_back(new Reservoir_MLP<xpu>(in_neurons, res_neurons, 3, mlp0, 0.85, "tanh"));
		this->layers.push_back(new Stacked_LSTM3<xpu>(base_neurons, 1000, base_neurons, lstm_layers));
		this->layers.push_back(new FullConnectLayer<xpu>(1000, out_neurons));
		this->layers.push_back(new FullConnectLayer<xpu>(1000, out_neurons));
//		this->layers.push_back(new FullConnectLayer<xpu>(1000, out_neurons));
//		this->layers.push_back((new GateLayer<xpu>("none", "sigmoid")));
		this->layers.push_back(new NonLinearLayer<xpu>("none"));
		this->layers.push_back((new GateLayer<xpu>("none", "sigmoid")));

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);

		/*
			0 ngram
			1 mlp0_out
			2 slstm_out
			3 out (5000)
			4 gate_ngram (5000)
			5 gate_slstm (5000)
		*/
		for(int i=0; i<6; i++){
			this->hidden_nodes.push_back(NULL);
		}
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;

		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		NgramNode<xpu> *tmpnode = new NgramNode<xpu>(in->N_all, in->N_batch, in->N_time, 5000);
		tmpnode->c_all = ((CharacterNode2<gpu> *)in)->c_all;
		this->hidden_nodes[0] = tmpnode;
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, base_neurons, false);
		this->hidden_nodes[2] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 1000, false);
		this->hidden_nodes[3] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, out_neurons, false);
		this->hidden_nodes[4] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, out_neurons, false);
		this->hidden_nodes[5] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, out_neurons, false);

		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[1]);
		this->layers[1]->set_node(this->hidden_nodes[1], this->hidden_nodes[2]);
		this->layers[2]->set_node(this->hidden_nodes[2], this->hidden_nodes[3]);
//		this->layers[3]->set_node(this->hidden_nodes[2], this->hidden_nodes[4]);
		this->layers[3]->set_node(this->hidden_nodes[2], this->hidden_nodes[5]);
//		this->layers[5]->set_node(this->hidden_nodes[0], this->hidden_nodes[4], this->out_nodes[0]);
		this->layers[4]->set_node(this->hidden_nodes[0], this->out_nodes[0]);
		this->layers[5]->set_node(this->hidden_nodes[3], this->hidden_nodes[5], this->out_nodes[0]);
	}
	void forward(int t=0){
		if(t==0){
			((NgramNode<gpu> *)this->hidden_nodes[0])->make_batch(((CharacterNode2<gpu> *)this->in_nodes[0])->lst);
		}
		for(int i=0; i<this->layers.size(); i++){
//			cerr << i << endl;
			this->layers[i]->forward(t);
		}
	}
};
/*
template<typename xpu>
class Hatsuwa_simple : public Layer<xpu>{
public:
	int inout_neurons=5000, base_neurons=500, res_neurons=500;
	Hatsuwa_simple(){
		int mlp0[5] = {5000, 400, 400, 400, 5000};

		this->layers.push_back(new MLP<xpu>(5, mlp0, "none"));

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
		// 0 short-circuit
		// 1 first reservoir
		// 2 common hidden
//		for(int i=0; i<3; i++){
//			this->hidden_nodes.push_back(NULL);
//		}
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
//
//		for(int i=0; i<this->hidden_nodes.size(); i++){
//			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
//		}
////		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 50, false);
	//	this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 500, false);
	//	this->hidden_nodes[2] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 500, false);

		this->layers[0]->set_node(this->in_nodes[0], this->out_nodes[0]);
	}
};
*/
template<typename xpu>
class Recall_long : public Layer<xpu>{
public:
	Recall_long(){
		int mlp[3] = {100, 100, 1};

		this->layers.push_back(new FullConnectLayer<xpu>(4, 100));
		this->layers.push_back(new NonLinearLayer<xpu>("tanh"));
		this->layers.push_back(new LSTM<xpu>(100,100,100));
		this->layers.push_back(new MLP<xpu>(3, mlp, "sigmoid"));

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);

		for(int i=0; i<2; i++){
			this->hidden_nodes.push_back(NULL);
		}
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;

		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 100, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, 100, false);

		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[0], this->hidden_nodes[0]);
		this->layers[2]->set_node(this->hidden_nodes[0], this->hidden_nodes[1]);
		this->layers[3]->set_node(this->hidden_nodes[1], this->out_nodes[0]);
	}
};
#endif
