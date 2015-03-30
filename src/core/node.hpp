#ifndef NODE_H
#define NODE_H

#include "mshadow/tensor.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <string>
#include <cassert>
#include <map>
using namespace std;
#include "util.hpp"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Node{
public:
	// T, Batch, dim
	TensorContainer<cpu, 3> x_all;  // full data (usually only input nodes)
	TensorContainer<cpu, 3> x_batch_cpu, dx_batch_cpu;  // working data @ cpu
	TensorContainer<xpu, 3> x_batch_gpu, dx_batch_gpu;  // working data @ gpu
	int N_all, N_batch, N_time, N_x;
	int set_zero_mode;  // 0: init to zero when batch changes
	                    // 1: バッチが変わって際に、最終時刻のデータを時刻0にコピー。残りは0。BPTTを途中で打ち切るときに有効
	Node(int n_all, int n_batch, int n_time, int n_x, bool is_data):N_all(n_all),N_batch(n_batch),N_time(n_time),N_x(n_x){
		if(Global_params::DEBUG) cerr << "new node(" << n_all << ", " << n_batch << ", " << n_x << ", " << is_data << ")" << endl;
		if(is_data) x_all.Resize(Shape3(1, N_all, N_x));
		x_batch_cpu.Resize(Shape3(N_time, N_batch, N_x));
		dx_batch_cpu.Resize(Shape3(N_time, N_batch, N_x));
		x_batch_gpu.Resize(Shape3(N_time, N_batch, N_x));
		dx_batch_gpu.Resize(Shape3(N_time, N_batch, N_x));

		set_zero_mode = 0;
	}
	~Node(){
		if(Global_params::DEBUG) cerr << "Delete Node "<< N_all << ", " << N_batch << ", " << N_x << endl;
	}

	virtual void set_zero(){
		if(Global_params::DEBUG) cerr << "set_zero nodes" << endl;
		if(set_zero_mode==1){
			Copy(x_batch_cpu[0], x_batch_gpu[0]);
			for(int t=1; t<N_time; t++){
				x_batch_cpu[t] = 0.0f;
			}
		}else{
			for(int t=0; t<N_time; t++){
				x_batch_cpu[t] = 0.0f;
			}
		}

		for(int t=0; t<N_time; t++){
			Copy(x_batch_gpu[t], x_batch_cpu[t]);
			dx_batch_cpu[t] = 0.0f;
			Copy(dx_batch_gpu[t], dx_batch_cpu[t]);
		}
		if(Global_params::DEBUG) cerr << "set_zero nodes end" << endl;
	}

	// lst: which data to use
	virtual void make_batch(int *lst){
		for(int t=0; t<N_time; t++){
			for(int i=0; i<N_batch; i++){
				Copy(x_batch_cpu[t][i], x_all[0][lst[i]+t]);
			}
			Copy(x_batch_gpu[t], x_batch_cpu[t]);
		}

	}

/*
	virtual void set_dx(Node<xpu> *target, string type){
		if(type=="mse"){
			calc_mse_loss<xpu>(this, target);
		}else if(type=="category"){
			calc_category_loss<xpu>(this, target);
		}else if(type=="nll"){
			calc_xe_loss<xpu>(this, target);
		}else{
			cerr << "invalid loss type" << endl;
		}
	}

	virtual float calc_error(Node<xpu> *target, string type){
		if(type=="mse"){
			return calc_mse_error<xpu>(this, target);
		}else if(type=="category"){
			return calc_category_error<xpu>(this, target);
		}else if(type=="nll"){
			return calc_log_likelihood<xpu>(this, target);
		}else{
			cerr << "invalid loss type" << endl;
		}
	}
	virtual void predict(string type){
		if(type=="mse"){

		}else if(type=="category"){
			for(int t=0; t<N_time; t++){
				Copy(x_batch_cpu[t], x_batch_gpu[t]);
				for(int i=0; i<N_batch; i++){
					float mx = x_batch_cpu[t][i][0];
					int mxi = 0;
					for(int j=0; j<N_x; j++){
						if(mx<x_batch_cpu[t][i][j]){
							mx = x_batch_cpu[t][i][j];
							mxi = j;
						}
					}
					for(int j=0; j<N_x; j++){
						if(mxi==j) x_batch_cpu[t][i][j]=1;
						else x_batch_cpu[t][i][j]=0;
					}
				}
//				Copy(x_batch_gpu[t], x_batch_cpu[t]);
			}
		}else if(type=="nll"){
			return calc_log_likelihood<xpu>(this, target);
		}else{
			cerr << "invalid loss type" << endl;
		}
	}
	virtual void sampling(string type, float beta=1.0){

	}
*/
};


// softmax loss
template<typename xpu>
void calc_category_loss(Node<xpu> *n0, Node<xpu> *n1){
	for(int t=0; t<n0->N_time; t++){
	    Tensor<xpu,1> tmp = NewTensor<xpu>(Shape1(n0->N_batch), 0.0f);
	    tmp = sumall_except_dim<0>(F<nl_exp>(n0->x_batch_gpu[t]));
	    float lambda = 0.0001;
	    n0->dx_batch_gpu[t] = n1->x_batch_gpu[t] - F<nl_exp>(n0->x_batch_gpu[t])/broadcast<0>(tmp, Shape2(n0->N_batch, n0->N_x));
	    n0->dx_batch_gpu[t] += lambda*(n0->N_x - broadcast<0>(tmp, Shape2(n0->N_batch, n0->N_x)))*F<nl_exp>(n0->x_batch_gpu[t]);
	}
}

// cross-entropy loss
template<typename xpu>
void calc_xe_loss(Node<xpu> *n0, Node<xpu> *n1){
	n0->dx_batch_gpu = F<xe_dx>(n0->x_batch_gpu, n1->x_batch_gpu);
}

template<typename xpu>
void calc_mse_loss(Node<xpu> *n0, Node<xpu> *n1){
	n0->dx_batch_gpu =  n1->x_batch_gpu - n0->x_batch_gpu;
}


template<typename xpu>
float calc_log_likelihood(Node<xpu> *n0, Node<xpu> *n1){
	float loss = 0;
	for(int t=0; t<n0->N_time; t++){
		n0->dx_batch_gpu[t] = F<xe_dx>(n0->x_batch_gpu[t], n1->x_batch_gpu[t]);
		TensorContainer<cpu, 1> tmp_cpu;
		TensorContainer<xpu, 1> tmp;
		tmp_cpu.Resize(Shape1(n0->N_batch));
		tmp.Resize(Shape1(n0->N_batch));
		tmp = sumall_except_dim<0>(F<xe_ll>(n0->x_batch_gpu[t], n1->x_batch_gpu[t]));
		Copy(tmp_cpu, tmp);
		for(int i=0; i<n0->N_batch; i++){
			loss += tmp_cpu[i];
		}
	}
	return loss/(float)n0->N_batch/(float)n0->N_time;
}



template<typename xpu>
float calc_accuracy(Node<xpu> *n0, Node<xpu> *n1){
	float loss = 0;
	for(int t=0; t<n0->N_time; t++){
//		cerr << t << endl;
	    Copy(n0->x_batch_cpu[t], n0->x_batch_gpu[t]);
	    for(int i=0; i<n0->N_batch; i++){
	        float mx = -100000;
	        int amx = 0;
	        for(int j=0; j<n0->N_x; j++){
	            if(mx<n0->x_batch_cpu[t][i][j]){
	                mx = n0->x_batch_cpu[t][i][j];
	                amx = j;
	            }
	        }
//	        cerr << t << " " << amx << " " << n0->x_batch_cpu[t][i][amx] << endl;
	        if(n1->x_batch_cpu[t][i][amx]>0.5){
	                loss+=1;
	        }
	    }
	}

    return loss/(float)n0->N_batch/(float)n0->N_time;
}


template<typename xpu>
float calc_mse_error(Node<xpu> *n0, Node<xpu> *n1){
	TensorContainer<cpu, 1> tmp_cpu;
	TensorContainer<xpu, 1> tmp;
	tmp_cpu.Resize(Shape1(n0->N_batch));
	tmp.Resize(Shape1(n0->N_batch));

	float loss = 0;
	for(int t=0; t<n0->N_time; t++){
		n0->dx_batch_gpu[t] =  n1->x_batch_gpu[t] - n0->x_batch_gpu[t];
		/*
		Copy(n0->x_batch_cpu[t], n0->x_batch_gpu[t]);
		for(int i=0; i<n0->N_x; i++){
			cerr << t << " " << i << " " << n0->x_batch_cpu[t][0][i] << " " << n1->x_batch_cpu[t][0][i] << endl;
		}
		*/

		tmp = sumall_except_dim<0>(F<square>(n0->dx_batch_gpu[t]));
		Copy(tmp_cpu, tmp);
		for(int i=0; i<n0->N_batch; i++){
//			cerr << tmp_cpu[i] << endl;
			loss += tmp_cpu[i];
		}
	}
//	n0->dx_batch_gpu;
	return loss/(float)n0->N_batch/(float)n0->N_time;

}

#endif