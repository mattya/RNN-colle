#ifndef LAYER_H
#define LAYER_H

#include "mshadow/tensor.h"
#include <iostream>
#include <vector>
using namespace std;
#include "util.hpp"
#include "node.hpp"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// Layer is connections from in_nodes to out_nodes
//    Nodes (in, hidden, out)
//    Layers (connection between nodes.  it may contain another layer)
template<typename xpu>
class Layer{
public:
	string layer_name;
	vector<Node<xpu> *> hidden_nodes, in_nodes, out_nodes;
	vector<Layer<xpu> *> layers;

// TBD multiple in, out nodes
	virtual void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
	}
	virtual void set_node(Node<xpu> *in, Node<xpu> *in2, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->in_nodes[1] = in2;
		this->out_nodes[0] = out;
	}
	virtual void forward(int t=0){
		for(int i=0; i<this->layers.size(); i++){
//			cerr << i << endl;
			this->layers[i]->forward(t);
		}
	}
	virtual void backward(int t=0){
		for(int i=this->layers.size()-1; i>=0; i--){
			this->layers[i]->backward(t);
		}
	}
	virtual void update(){
		for(int i=0; i<this->layers.size(); i++){
			this->layers[i]->update();
		}
	}
	virtual void set_zero(){
		for(int i=0; i<this->layers.size(); i++){
			if(Global_params::DEBUG) cerr << "set_zero layers " << i << endl;
			this->layers[i]->set_zero();
		}

		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(Global_params::DEBUG) cerr << "set_zero hidden " << i << endl;
			this->hidden_nodes[i]->set_zero();
		}
		for(int i=0; i<this->out_nodes.size(); i++){
			if(Global_params::DEBUG) cerr << "set_zero out " << i << endl;
			this->out_nodes[i]->set_zero();
		}
	}

	virtual void set_param(string param_name, float value){
		for(int i=0; i<this->layers.size(); i++){
			this->layers[i]->set_param(param_name, value);
		}
	}
	virtual float get_param(string param_name){
		if(this->layers.size()==0) return 0;
		else return this->layers[0]->get_param(param_name);
	}

	virtual void save_model(FILE *fo){
		for(int i=0; i<this->layers.size(); i++){
			this->layers[i]->save_model(fo);
		}
	}

	virtual void load_model(FILE *fi){
		for(int i=0; i<this->layers.size(); i++){
			this->layers[i]->load_model(fi);
		}
	}

	~Layer(){
		for(int i=0; i<this->hidden_nodes.size(); i++){
			delete this->hidden_nodes[i];
		}
	}
};

// y = Wx + b
// dense linear transformation layer
template<typename xpu>
class FullConnectLayer : public Layer<xpu>{
public:
	Random<xpu, float> rnd;
	TensorContainer<xpu, 1, float> b, g_b;
	TensorContainer<xpu, 2, float> w, g_w, rms_w, mask, dropout_cpy;
	float eta, decay, momentum;

	string sgd_algo;

	float lr_mult;
	int lr_mult_interval;
	float base_lr;
	float current_lr;

	bool is_dropout;

	FullConnectLayer(int n_in, int n_out, float scale=0.99):rnd(0){
		w.Resize(Shape2(n_out, n_in));
		g_w.Resize(Shape2(n_out, n_in));
		rms_w.Resize(Shape2(n_out, n_in));
		b.Resize(Shape1(n_out));
		g_b.Resize(Shape1(n_out));

/*
		// init matrix (xavier initialize)
		float a;
		a = sqrt(3.0f / (n_out + n_in));
		rnd.SampleUniform(w, -a, a);
*/
		// init matrix (orthogonal initialize)
		if(Global_params::init_flag){
			random_orthogonal<xpu>(w, scale, 1.0f);
			rnd.SampleUniform(&b, -0.01f, 0.01f);
		}
//		Copy(b, init_b);

		momentum = 0.95;
		rms_w = 0;
//		rms_w += 0.001;
		g_w = 0;
		g_b = 0;
		eta = 0.001;
		decay = 0.00001;

		is_dropout = false;

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
	}
	~FullConnectLayer(){

	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;

		if(is_dropout & Global_params::phase_train){
			dropout_cpy.Resize(in->x_batch_gpu[0].shape_);
			mask.Resize(Shape2(in->N_batch, in->N_x));
		}
	}

	void forward(int t=0){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];
		int N_batch = in->N_batch;
		if(is_dropout & Global_params::phase_train){
			mask = rnd.uniform(mask.shape_);
			mask = F<dropout>(0.5f, mask);
			Copy(dropout_cpy, in->x_batch_gpu[t]);
			dropout_cpy *= mask;
			out->x_batch_gpu[t] += dot(dropout_cpy, w.T());
		}else{
			out->x_batch_gpu[t] += dot(in->x_batch_gpu[t], w.T());
		}
		out->x_batch_gpu[t] += repmat(b, N_batch);
	}
	void backward(int t=0){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];
		if(is_dropout & Global_params::phase_train){
			dropout_cpy = dot(out->dx_batch_gpu[t], w);
			dropout_cpy *= mask;
			in->dx_batch_gpu[t] += dropout_cpy;
		}else{
			in->dx_batch_gpu[t] += dot(out->dx_batch_gpu[t], w);
		}
	}

	void update(){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];

		out->dx_batch_gpu = F<clip>(out->dx_batch_gpu);

		if(sgd_algo=="rmsprop"){
			g_b *= momentum;
//			cerr << out->dx_batch_gpu.shape_[0] << ", " << out->dx_batch_gpu.shape_[1] << ", " << out->dx_batch_gpu.shape_[2] << ", " << out->dx_batch_gpu.shape_[3] << endl;
//			cerr << "hoge" << endl;
			 g_b += (1.0-momentum)*sumall_except_dim<2>(out->dx_batch_gpu);

//			cerr << "piyo" << endl;
			 g_w = 0;
			 for(int t=0; t<in->N_time; t++)
				 g_w += dot(out->dx_batch_gpu[t].T(), in->x_batch_gpu[t]);
			 rms_w *= momentum;
			 rms_w += (1.0-momentum)*F<square>(g_w);
			 g_w = F<inv_sqrt>(g_w, rms_w);
//			cerr << "fuga" << endl;
		}else if(sgd_algo=="momentum"){
			g_b *= momentum;
			g_w *= momentum;
			for(int t=0; t<in->N_time; t++){
				g_b += (1.0-momentum)*sum_rows(out->dx_batch_gpu[t]);
				g_w += (1.0-momentum)*dot(out->dx_batch_gpu[t].T(), in->x_batch_gpu[t]);
			}
		}else{
			cerr << "invalid sgd algorithm: " <<  sgd_algo << endl;
		}

		b += eta*g_b;
		w += eta*(-decay*w + g_w);
	}

	void set_zero(){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];
		// in->set_zero();
		out->set_zero();
	}

	void set_param(string param_name, float value){
		if(param_name=="eta"){
			eta = value;
		}else if(param_name=="decay"){
			decay = value;
		}else if(param_name=="momentum"){
			momentum = value;
		}else if(param_name=="is_dropout"){
			is_dropout = (bool)value;
		}else if(param_name=="sgd_algo"){
			sgd_algo = value==0?"momentum":"rmsprop";
		}
	}
	float get_param(string param_name){
		if(param_name=="eta"){
			return eta;
		}else if(param_name=="decay"){
			return decay;
		}else if(param_name=="momentum"){
			return momentum;
		}
		return 0;
	}

	void save_model(FILE *fo){
		cerr << "save: " << b.size(0) << " " << w.size(0) << "," << w.size(1) << endl;
		FileStream fs(fo);
		b.SaveBinary(fs);
		w.SaveBinary(fs);
	}

	void load_model(FILE *fi){
		cerr << "load: " << b.size(0) << " " << w.size(0) << "," << w.size(1) << endl;
		FileStream fs(fi);
		b.LoadBinary(fs);
		w.LoadBinary(fs);
	}
};

// NonLinearLayer(type)
// type: "tanh", "sigmoid", "relu", "none"
template<typename xpu>
class NonLinearLayer : public Layer<xpu>{
public:
	string type;
	NonLinearLayer(string t0):type(t0){
		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
	}
	~NonLinearLayer(){
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
	}
	void forward(int t=0){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];
		if(type=="tanh"){
			out->x_batch_gpu[t] = F<nl_tanh>(in->x_batch_gpu[t]);
		}else if(type=="sigmoid"){
			out->x_batch_gpu[t] = F<nl_sigmoid>(in->x_batch_gpu[t]);
		}else if(type=="relu"){
			out->x_batch_gpu[t] = F<nl_relu>(in->x_batch_gpu[t]);
		}else if(type=="none"){
			Copy(out->x_batch_gpu[t], in->x_batch_gpu[t]);
		}
	}
	void backward(int t=0){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];
		if(type=="tanh"){
			out->dx_batch_gpu[t] *= F<nl_dtanh>(out->x_batch_gpu[t]);
		}else if(type=="sigmoid"){
			out->dx_batch_gpu[t] *= F<nl_dsigmoid>(out->x_batch_gpu[t]);
		}else if(type=="relu"){
			out->dx_batch_gpu[t] *= F<nl_drelu>(out->x_batch_gpu[t]);
		}else if(type=="none"){

		}
		Copy(in->dx_batch_gpu[t], out->dx_batch_gpu[t]);
	}
};

// this layer causes 1 step delay
// most important part of RNN
template<typename xpu>
class TimeShiftLayer : public Layer<xpu>{
public:
	TimeShiftLayer(){
		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
	}
	~TimeShiftLayer(){
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
	}
	void forward(int t=0){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];
		if(t==this->in_nodes[0]->N_time-1) Copy(out->x_batch_gpu[0], in->x_batch_gpu[t]);
		else Copy(out->x_batch_gpu[t+1], in->x_batch_gpu[t]);
//		cerr << t << " " << this->in_nodes[0]->N_batch-1 << endl;
	}
	void backward(int t=0){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];
//		if(t==this->in_nodes[0]->N_batch-1) in->dx_batch_gpu[t] += out->dx_batch_gpu[0];
		if(t==this->in_nodes[0]->N_time-1) ;
		else in->dx_batch_gpu[t] += out->dx_batch_gpu[t+1];
//		else in->dx_batch_gpu[t] = 0;
	}
};

// GateLayer(nl0, nl1)
// nl0, nl1: nonlinear type ("tanh", "sigmoid", "relu", "none")
//
// in0 -> nl0 --|
// in1 -> nl1 - * -> out
//
template<typename xpu>
class GateLayer : public Layer<xpu>{
public:
	GateLayer(string nl0="tanh", string nl1="sigmoid"){
		this->in_nodes.push_back(NULL);
		this->in_nodes.push_back(NULL);
		this->hidden_nodes.push_back(NULL);
		this->hidden_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);

		this->layers.push_back(new NonLinearLayer<xpu>(nl0));
		this->layers.push_back(new NonLinearLayer<xpu>(nl1));
	}
	~GateLayer(){
	}
	void set_node(Node<xpu> *in0, Node<xpu> *in1, Node<xpu> *out){
		this->in_nodes[0] = in0;
		this->in_nodes[1] = in1;
		this->out_nodes[0] = out;
		for(int i=0; i<2; i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		this->hidden_nodes[0] = new Node<xpu>(in0->N_all, in0->N_batch, in0->N_time, in0->N_x, false);
		this->hidden_nodes[1] = new Node<xpu>(in1->N_all, in1->N_batch, in0->N_time, in1->N_x, false);

		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->in_nodes[1], this->hidden_nodes[1]);
	}
	void forward(int t=0){
		Node<xpu> *in = this->in_nodes[0];
		Node<xpu> *out = this->out_nodes[0];
		for(int i=0; i<this->layers.size(); i++){
			this->layers[i]->forward(t);
		}
//		cerr << this->out_nodes[0]->N_x << this->hidden_nodes[0]->N_x << this->hidden_nodes[1]->N_x << endl;
		this->out_nodes[0]->x_batch_gpu[t] += this->hidden_nodes[0]->x_batch_gpu[t] * this->hidden_nodes[1]->x_batch_gpu[t];
	}
	void backward(int t=0){
		this->hidden_nodes[0]->dx_batch_gpu[t] += this->out_nodes[0]->dx_batch_gpu[t] * this->hidden_nodes[1]->x_batch_gpu[t];
		this->hidden_nodes[1]->dx_batch_gpu[t] += this->out_nodes[0]->dx_batch_gpu[t] * this->hidden_nodes[0]->x_batch_gpu[t];
		for(int i=this->layers.size()-1; i>=0; i--){
			this->layers[i]->backward(t);
		}
	}
};

// Multi Layer Perceptron
//
// MLP(num_layers, *num_x, out_nl, hidden_nl)
// num_layers: number of layers
// num_x[num_layers]: number of neurons per each layer
// out_nl: nonlinear type @ last layer
// hidden_nl: nonlinear type @ middle layers
template<typename xpu>
class MLP : public Layer<xpu>{
public:
	int num_layers;
	int *nodes_per_layers;
	MLP(int num_layers_, int *num_x, string out_nl="none", string hidden_nl="tanh"):num_layers(num_layers_){
		nodes_per_layers = (int *)malloc(num_layers*sizeof(int));
		for(int i=0; i<num_layers; i++) nodes_per_layers[i] = num_x[i];
		for(int i=0; i<num_layers-1; i++){
			this->layers.push_back((new FullConnectLayer<xpu>(nodes_per_layers[i], nodes_per_layers[i+1])));
			if(i<num_layers-2){
				this->layers.push_back((new NonLinearLayer<xpu>(hidden_nl)));
			}

		}
		this->layers.push_back((new NonLinearLayer<xpu>(out_nl)));

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
		for(int i=0; i<num_layers-2; i++){
			this->hidden_nodes.push_back(NULL);
		}

	}
	// len(this->hidden_nodes) == num_layers-2
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
		for(int i=0; i<num_layers-2; i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
			this->hidden_nodes[i] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_per_layers[i+1], false);
		}
		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[0], this->hidden_nodes[0]);
		for(int i=0; i<num_layers-3; i++){
			this->layers[2*i+2]->set_node(this->hidden_nodes[i], this->hidden_nodes[i+1]);
//			this->layers[2*i+2]->set_param("is_dropout", true);
			this->layers[2*i+3]->set_node(this->hidden_nodes[i+1], this->hidden_nodes[i+1]);
		}
		this->layers[2*num_layers-4]->set_node(this->hidden_nodes[num_layers-3], this->out_nodes[0]);
//		this->layers[2*num_layers-4]->set_param("is_dropout", true);
		this->layers[2*num_layers-3]->set_node(this->out_nodes[0], this->out_nodes[0]);
	}
};

// Reservoir Computing
// Reservoir(n_in, n_res, n_out, scale)
// learn only res-out connection
template<typename xpu>
class Reservoir : public Layer<xpu>{
public:
	int nodes_in, nodes_res, nodes_out;
	Reservoir(int n_in, int n_res, int n_out, float scale=0.9){
		nodes_in = n_in;
		nodes_res = n_res;
		nodes_out = n_out;

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_res)));
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_res, nodes_res, scale)));
		this->layers.push_back((new TimeShiftLayer<xpu>()));
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_res, nodes_out)));

		this->in_nodes.push_back(NULL);
		this->hidden_nodes.push_back(NULL);  // t
		this->hidden_nodes.push_back(NULL);  // t+1
		this->out_nodes.push_back(NULL);

	}

	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
		if(this->hidden_nodes[1]) delete this->hidden_nodes[1];
		if(this->hidden_nodes[0]) delete this->hidden_nodes[0];
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[1], this->hidden_nodes[0]);
		this->layers[2]->set_node(this->hidden_nodes[0], this->hidden_nodes[1]);
		this->layers[3]->set_node(this->hidden_nodes[0], this->out_nodes[0]);
//		this->layers[4]->set_node(this->out_nodes[0], this->out_nodes[0]);

		this->hidden_nodes[1]->set_zero();
		this->hidden_nodes[1]->set_zero_mode = Global_params::handover;
	}

	// no need to backprop
	void backward(int t=0){
	}
	void update(){
		this->layers[3]->update();
	}
};

// Reservoir Computing
// Reservoir(n_in, n_res, n_out, num_layers, num_x, scale, out_nl, hidden_nl)
// learn only res-out connection
template<typename xpu>
class Reservoir_MLP : public Layer<xpu>{
public:
	int nodes_in, nodes_res, nodes_out;
	Reservoir_MLP(int n_in, int n_res, int n_out, int num_layers_, int *num_x, float scale=0.9, string out_nl="none", string hidden_nl="tanh"){
		nodes_in = n_in;
		nodes_res = n_res;
		nodes_out = n_out;

		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_res)));
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_res, nodes_res, scale)));
		this->layers.push_back((new TimeShiftLayer<xpu>()));
		this->layers.push_back((new MLP<xpu>(num_layers_, num_x, out_nl, hidden_nl)));

		this->in_nodes.push_back(NULL);
		this->hidden_nodes.push_back(NULL);  // t
		this->hidden_nodes.push_back(NULL);  // t+1
		this->out_nodes.push_back(NULL);

	}

	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
		if(this->hidden_nodes[1]) delete this->hidden_nodes[1];
		if(this->hidden_nodes[0]) delete this->hidden_nodes[0];
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_res, false);
		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[1], this->hidden_nodes[0]);
		this->layers[2]->set_node(this->hidden_nodes[0], this->hidden_nodes[1]);
		this->layers[3]->set_node(this->hidden_nodes[0], this->out_nodes[0]);
//		this->layers[4]->set_node(this->out_nodes[0], this->out_nodes[0]);

		this->hidden_nodes[1]->set_zero();
		this->hidden_nodes[1]->set_zero_mode = Global_params::handover;
	}

	// no need to backprop
	void backward(int t=0){
	}
	void update(){
		this->layers[3]->update();
	}
};

// LSTM(n_in, n_cell, n_out)
/*
	in(t), out(t-1) -> i, g, f, o
	g, i<gate> -> cell
	cell(t-1), f<gate> -> cell
	cell(t), o<gate> -> out(t)

	assume n_cell == n_in == n_out
*/
template<typename xpu>
class LSTM : public Layer<xpu>{
public:
	int nodes_in, nodes_cell, nodes_out, nodes_hidden;
	LSTM(int n_in, int n_cell, int n_out){
		nodes_in = n_in;
		nodes_cell = n_cell;
		nodes_hidden = nodes_cell;
		nodes_out = n_out;


		// i
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_cell)));  // 0
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_hidden, nodes_cell))); // 1

		// g
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_cell)));  // 2
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_hidden, nodes_cell))); // 3

		// f
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_cell)));  // 4
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_hidden, nodes_cell))); // 5

		// o
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_in, nodes_hidden)));  // 6
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_hidden, nodes_hidden))); // 7

		// gates
		this->layers.push_back((new GateLayer<xpu>("tanh", "sigmoid")));        // 8
		this->layers.push_back((new GateLayer<xpu>("none", "sigmoid")));        // 9
		this->layers.push_back((new GateLayer<xpu>("tanh", "sigmoid")));        // 10

		// timeshifts
		this->layers.push_back((new TimeShiftLayer<xpu>()));   // 11
		this->layers.push_back((new TimeShiftLayer<xpu>()));   // 12

		// out
		this->layers.push_back((new FullConnectLayer<xpu>(nodes_hidden, nodes_out)));  // 0

		this->in_nodes.push_back(NULL);
		this->hidden_nodes.push_back(NULL);  // 0 i
		this->hidden_nodes.push_back(NULL);  // 1 g
		this->hidden_nodes.push_back(NULL);  // 2 f
		this->hidden_nodes.push_back(NULL);  // 3 o
		this->hidden_nodes.push_back(NULL);  // 4 cell(t)
		this->hidden_nodes.push_back(NULL);  // 5 cell(t+1)
		this->hidden_nodes.push_back(NULL);  // 6 hidden(t)
		this->hidden_nodes.push_back(NULL);  // 7 hidden(t+1)
		this->out_nodes.push_back(NULL);     // out

	}

	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;
//		if(this->out_nodes[1]) delete this->out_nodes[1];
//		this->out_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_out, false);
		for(int i=0; i<8; i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
		}
		this->hidden_nodes[0] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_cell, false);
		this->hidden_nodes[1] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_cell, false);
		this->hidden_nodes[2] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_cell, false);
		this->hidden_nodes[3] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_hidden, false);
		this->hidden_nodes[4] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_cell, false);
		this->hidden_nodes[5] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_cell, false);
		this->hidden_nodes[6] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_hidden, false);
		this->hidden_nodes[7] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, nodes_hidden, false);

		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[7], this->hidden_nodes[0]);
		this->layers[2]->set_node(this->in_nodes[0], this->hidden_nodes[1]);
		this->layers[3]->set_node(this->hidden_nodes[7], this->hidden_nodes[1]);
		this->layers[4]->set_node(this->in_nodes[0], this->hidden_nodes[2]);
		this->layers[5]->set_node(this->hidden_nodes[7], this->hidden_nodes[2]);
		this->layers[6]->set_node(this->in_nodes[0], this->hidden_nodes[3]);
		this->layers[7]->set_node(this->hidden_nodes[7], this->hidden_nodes[3]);

		this->layers[8]->set_node(this->hidden_nodes[1], this->hidden_nodes[0], this->hidden_nodes[4]);
		this->layers[9]->set_node(this->hidden_nodes[5], this->hidden_nodes[2], this->hidden_nodes[4]);
		this->layers[10]->set_node(this->hidden_nodes[4], this->hidden_nodes[3], this->hidden_nodes[6]);
		this->layers[11]->set_node(this->hidden_nodes[4], this->hidden_nodes[5]);
		this->layers[12]->set_node(this->hidden_nodes[6], this->hidden_nodes[7]);
		this->layers[13]->set_node(this->hidden_nodes[6], this->out_nodes[0]);

		this->hidden_nodes[5]->set_zero();
		this->hidden_nodes[7]->set_zero();
		this->hidden_nodes[5]->set_zero_mode = Global_params::handover;
		this->hidden_nodes[7]->set_zero_mode = Global_params::handover;
	}
};


// Stacked_LSTM(n_in, n_out, n_base, lstm_layers)
// n_in: input neurons
// n_out: output neurons
// n_base: hidden/cell neurons of LSTM
// lstm_layers: number of LSTM layers

// in +-> lstm0 -+
//    |     |    |
//    +-> lstm1 -+
//    |     |    |
//    +    ...   +
//    |     |    |
//    +-> lstmN -+->out

template<typename xpu>
class Stacked_LSTM : public Layer<xpu>{
public:
	int num_layers;
	int in_neurons, out_neurons, base_neurons;
	Stacked_LSTM(int in_neurons_, int out_neurons_, int base_neurons_, int lstm_layers){
		in_neurons = in_neurons_;
		out_neurons = out_neurons_;
		base_neurons = base_neurons_;
		num_layers = lstm_layers;

		for(int i=0; i<num_layers; i++){
			this->layers.push_back(new FullConnectLayer<xpu>(in_neurons, base_neurons));
			this->layers.push_back(new LSTM<xpu>(base_neurons,base_neurons,base_neurons));
			this->layers.push_back(new FullConnectLayer<xpu>(base_neurons, out_neurons));
		}

		this->in_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
		for(int i=0; i<num_layers+1; i++){
			// 0 lstm1_in
			// 1 lstm2_in, lstm1_out
			// 2 lstm2_out
			this->hidden_nodes.push_back(NULL);
		}
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;

		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
			this->hidden_nodes[i] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, base_neurons, false);
		}

		for(int i=0; i<num_layers; i++){
			this->layers[3*i+0]->set_node(this->in_nodes[0], this->hidden_nodes[i]);
			this->layers[3*i+1]->set_node(this->hidden_nodes[i], this->hidden_nodes[i+1]);
			this->layers[3*i+2]->set_node(this->hidden_nodes[i], this->out_nodes[0]);
		}
	}

};

template<typename xpu>
class Stacked_LSTM_MLP : public Layer<xpu>{
public:
	int num_layers;
	int in_neurons, mid_neurons, out_neurons, base_neurons;
	Stacked_LSTM_MLP(int in_neurons_, int base_neurons_, int mid_neurons_, int out_neurons_, int lstm_layers, int mlp_layers, int *mlp_xs, string out_nl="none"){
		in_neurons = in_neurons_;
		mid_neurons = mid_neurons_;
		out_neurons = out_neurons_;
		base_neurons = base_neurons_;
		num_layers = lstm_layers;

		this->layers.push_back(new Stacked_LSTM<xpu>(in_neurons, mid_neurons, base_neurons, lstm_layers));
		this->layers.push_back(new MLP<xpu>(mlp_layers, mlp_xs, out_nl));


		this->in_nodes.push_back(NULL);
		this->hidden_nodes.push_back(NULL);
		this->out_nodes.push_back(NULL);
	}
	void set_node(Node<xpu> *in, Node<xpu> *out){
		this->in_nodes[0] = in;
		this->out_nodes[0] = out;

		for(int i=0; i<this->hidden_nodes.size(); i++){
			if(this->hidden_nodes[i]) delete this->hidden_nodes[i];
			this->hidden_nodes[i] = new Node<xpu>(in->N_all, in->N_batch, in->N_time, mid_neurons, false);
		}

		this->layers[0]->set_node(this->in_nodes[0], this->hidden_nodes[0]);
		this->layers[1]->set_node(this->hidden_nodes[0], this->out_nodes[0]);
	}

};
#endif