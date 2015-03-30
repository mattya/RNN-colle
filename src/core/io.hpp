#ifndef IO_H
#define IO_H

#include "node.hpp"
#include "util.hpp"

#include <cstdio>
#include <iostream>
#include <string>
using namespace std;

typedef unsigned char byte;


/*
	data[time][data][dim]
*/
template<typename xpu>
void read_float_binary_data(string filename, Node<xpu> *node, int num_data, int data_time, int data_dim){
	cerr << "read_float_binary_data from " << filename << endl;
	float *data = (float *)malloc(num_data*data_dim*sizeof(float));
	FILE *fp = fopen(filename.c_str(), "rb");
	fread(data, sizeof(float), num_data*data_dim, fp);
	fclose(fp);

	for(int i=0; i<num_data; ++i){
//		for(int t=0; t<data_time; ++t){
			for(int j=0; j<data_dim; ++j){
				node->x_all[0][i][j] = (float)data[i*data_dim+j];
	//			cerr << node.x_all[i][j] << endl;
//				node->x_all[t][i][j] = node->x_all[t][i][j];
			}
//		}
	}
	free(data);
}

/*
	data[time][data][dim]
*/
template<typename xpu>
void read_int_binary_data(string filename, Node<xpu> *node, int num_data, int data_time, int data_dim){
	cerr << "read_int_binary_data from " << filename << endl;
	int *data = (int *)malloc(num_data*data_dim*sizeof(int));
	FILE *fp = fopen(filename.c_str(), "rb");
//	cerr << num_data*data_dim << endl;
	fread(data, sizeof(int), num_data*data_dim, fp);
	fclose(fp);
	for(int i=0; i<num_data; ++i){
//		for(int t=0; t<data_time; ++t){
			for(int j=0; j<data_dim; ++j){
				node->x_all[0][i][j] = (float)data[i*data_dim+j];
//				cerr << node->x_all[0][i][j] << endl;
//				node->x_all[t][i][j] = node->x_all[t][i][j];
			}
//		}
	}
	free(data);
}


template<typename xpu>
void read_mnist_data(string filename, Node<xpu> *node, int num_data, int data_dim, int skip_byte, float mean=128.0, float scale=1.0/128.0){
	cerr << "read mnist data from " << filename << endl;
	byte *data = (byte *)malloc(num_data*data_dim*sizeof(byte));
	FILE *fp = fopen(filename.c_str(), "rb");
	fseek(fp, skip_byte, SEEK_CUR);
	fread(data, sizeof(byte), num_data*data_dim, fp);
	fclose(fp);

	for(int i=0; i<num_data; ++i){
		for(int j=0; j<data_dim; ++j){
			node->x_all[0][i][j] = (float)data[i*data_dim+j];
//			cerr << node->x_all[0][i][j] << endl;
			node->x_all[0][i][j] = (node->x_all[0][i][j]-mean)*scale;
		}
	}
}


template<typename xpu>
void read_mnist_label(string filename, Node<xpu> *node, int num_data, int data_dim, int skip_byte){
	cerr << "read mnist label from " << filename << endl;
	byte *data = (byte *)malloc(num_data*data_dim*sizeof(byte));
	FILE *fp = fopen(filename.c_str(), "rb");
	fseek(fp, skip_byte, SEEK_CUR);
	fread(data, sizeof(byte), num_data*1, fp);
	fclose(fp);

	// size 0, 1 correct?
	for(int i=0; i<node->x_all.size(1); ++i){
		int label = (int)data[i];
		for(int j=0; j<node->x_all.size(2); ++j){
			node->x_all[0][i][j] = (j==label?1:0);
//			cerr << j << " " << node->x_all[0][i][j] << endl;
		}
	}
}


/*
template<typename xpu>
void read_float_binary_data(string filename, Node<xpu> *node, int num_data, int data_dim, int skip_byte, float mean=0, float scale=1.0){
	cerr << "read float binary data from " << filename << endl;
//	assert(node->N_time==1);
	float *data = (float *)malloc(num_data*data_dim*sizeof(float));
	FILE *fp = fopen(filename.c_str(), "rb");
	fseek(fp, skip_byte, SEEK_CUR);
	fread(data, sizeof(float), num_data*data_dim, fp);
	fclose(fp);
	for(int i=0; i<num_data; ++i){
		for(int j=0; j<data_dim; ++j){
			node.x_all[0][i][j] = (float)data[i*data_dim+j];
//			cerr << node.x_all[i][j] << endl;
			node.x_all[0][i][j] = (node.x_all[0][i][j]-mean)*scale;
		}
	}
	free(data);
}*/

template<typename xpu>
void read_float_data(string filename, Node<xpu> *node, int num_data, int num_t, int data_dim, int skip_num=0, float mean=0, float scale=1.0){
	cerr << "read float data from " << filename << endl;
//	assert(node.N_time==1);
	float *data = (float *)malloc(num_data*data_dim*sizeof(float));
	FILE *fp = fopen(filename.c_str(), "r");
	float gomi;
	for(int i=0; i<skip_num*data_dim; i++) fscanf(fp, "%f", &gomi);
	for(int i=0; i<num_data*data_dim; i++) fscanf(fp, "%f", &data[i]);
	fclose(fp);

	int cnt = 0;
	for(int i=0; i<num_data; ++i){
//		for(int t=0; t<num_t; ++t){
			for(int j=0; j<data_dim; ++j){
//				cerr << cnt << endl;
				node->x_all[0][i][j] = (float)data[cnt++];
	//			cerr << node.x_all[i][j] << endl;
				node->x_all[0][i][j] = (node->x_all[0][i][j]-mean)*scale;
			}
//		}
	}
	free(data);
}


template<typename xpu>
void write_float_data(string filename, Node<xpu> *node, int num_data, int data_dim, int skip_num=0, float mean=0, float scale=1.0){
	cerr << "write float data to " << filename << endl;
//	assert(node.N_time==1);

	FILE *fp = fopen(filename.c_str(), "w");
	for(int i=0; i<num_data; ++i){
		for(int j=0; j<data_dim; ++j){
			fprintf(fp, "%f ", scale*(node->x_all[0][i][j]-mean));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

#endif
