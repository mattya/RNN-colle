#ifndef UTIL_HPP
#define UTIL_HPP


#include "mshadow/tensor.h"
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

class Global_params{
public:
	static string python;
	static string prefix;
	static bool phase_train;
	static bool DEBUG;
	static bool init_flag;
	static bool handover;
};
string Global_params::python = "python";
string Global_params::prefix = "test";
bool Global_params::phase_train = false;
bool Global_params::DEBUG = true;
bool Global_params::init_flag = true;
bool Global_params::handover = false;

// define tanh operation
struct nl_tanh{
    MSHADOW_XINLINE static float Map(float a) {
//    	return a>0?a:0;
        return  tanhf(a);
    }
};
struct nl_dtanh{
    MSHADOW_XINLINE static float Map(float a) {
//    	return a>0?1:0;
        return  (1.0f-a)*(1.0f+a);
    }
};
struct nl_sigmoid{
    MSHADOW_XINLINE static float Map(float a) {
//    	return a>0?a:0;
        return 1.0f/(1.0f+expf(-a));
    }
};
struct nl_dsigmoid{
    MSHADOW_XINLINE static float Map(float a) {
//    	return a>0?1:0;
        return  (1.0f-a)*a;
    }
};
struct nl_relu{
    MSHADOW_XINLINE static float Map(float a) {
    	return a>0?a:0;
    }
};
struct nl_drelu{
    MSHADOW_XINLINE static float Map(float a) {
    	return a>0?1:0;
    }
};
struct nl_exp{
    MSHADOW_XINLINE static float Map(float a) {
//    	return a>0?a:0;
        return  expf(a);
    }
};
struct xe_dx{
	MSHADOW_XINLINE static float Map(float a, float b){
		return (b-a)/(a*(1.0f-a)+1e-6);
	}
};
struct xe_ll{
	MSHADOW_XINLINE static float Map(float a, float b){
		return b>0.5f?logf(a+1e-10):logf(1.0f-a+1e-10);
	}
};
struct square{
    MSHADOW_XINLINE static float Map(float a) {
        return  a*a;

    }
};
struct clip{
    MSHADOW_XINLINE static float Map(float a) {
        return  a>10.0f?10.0f:(a<-10.0f?-10.0f:a);

    }
};
struct inv_sqrt{
    MSHADOW_XINLINE static float Map(float a, float b) {
        return a/(sqrt(b)+0.0001f);
    }
};
struct dropout{
	// p: prob to dropout
    MSHADOW_XINLINE static float Map(float p, float r) {
        if(p>r) return 0.0f;
        else return 1.0f/(1.0f-p);
    }
};

string to_string(const char* str, int i){
	char ret[100];
	sprintf(ret, "%s_%d", str, i);
	return string(ret);
}
string to_string(string str, int i){
	char ret[100];
	sprintf(ret, "%s_%d", str.c_str(), i);
	return string(ret);
}
/*
template<typename xpu, int dim0>
void save_binary(TensorContainer<xpu, dim0> &t0, FILE *fp){
	int dim = dim0;
	TensorContainer<cpu, dim0> t;
	t.Resize(t0.shape);
	Copy(t, t0);
	float *tmp;
	if(dim==1) tmp = new float[t0.shape[0]];

	fwrite(&dim, sizeof(int), 1, fp);
	for(int i=0; i<dim; i++) fwrite(&t.shape[i], sizeof(int), 1, fp);
	if(dim==1){
		for(int i=0; i<t.shape[0]; i++){
			fwrite(&t[i], sizeof(int), 1, fp);
		}
	}else if(dim==2){
		for(int i=0; i<t.shape[1]; i++){
			for(int j=0; j<t.shape[0]; j++){
				fwrite(&t[i][j], sizeof(int), 1, fp);
			}
		}
	}
}

template<typename xpu, int dim>
void load_binary(TensorContainer<xpu, dim0> &t, FILE *fp){
	int dim = dim0;
	TensorContainer<cpu, dim0> t;
	t.Resize(t0.shape);
	int d;
	fread(&d, sizeof(int), 1, fp);
	assert(d==dim);
	int s[4];
	for(int i=0; i<dim; i++){
		fread(&s[i], sizeof(int), 1, fp);
		assert(s[i]==t.shape[i]);
	}
	if(dim==1){
		for(int i=0; i<t.shape[0]; i++){
			fread(&t[i], sizeof(int), 1, fp);
		}
	}else if(dim==2){
		for(int i=0; i<t.shape[1]; i++){
			for(int j=0; j<t.shape[0]; j++){
				fread(&t[i][j], sizeof(int), 1, fp);
			}
		}
	}
	Copy(t0, t);
}
*/

/* from old mshadow */
class FileStream: public IStream{
public:
    /*! \brief constructor */
    FileStream( FILE *fp ):fp_(fp){}
    virtual size_t Read( void *ptr, size_t size ){
        return fread( ptr, size, 1, fp_ );
    }
    virtual void Write( const void *ptr, size_t size ){
        fwrite( ptr, size, 1, fp_ );
    }
    /*! \brief close file */
    inline void Close( void ){
        fclose( fp_ );
    }
private:
    FILE *fp_;
};

FileStream to_istream(string filename, const char* flag){
	FILE *f = fopen(filename.c_str(), flag);
	FileStream fs(f);
	return fs;
}


float random(float x0, float x1){
	return (x1-x0)*((float)rand()/(RAND_MAX+0.000001)) + x0;
}

void random_orthogonal_(float *W, int nrow, int ncol, float tauinv){
	char command[1000];
//	sprintf(command, "/opt/anaconda3/bin/python ./ipynb/svd.py %d %d %f ./tmp/w.txt", nrow, ncol, tauinv);
	sprintf(command, "%s ./tools/svd.py %d %d %f ./tmp/%s_w.txt", Global_params::python.c_str(), nrow, ncol, tauinv, Global_params::prefix.c_str());
	cerr << command << endl;
	system(command);

	string fn = "./tmp/"+Global_params::prefix+"_w.txt";
	FILE *f = fopen(fn.c_str(), "r");
	for(int i=0; i<nrow; i++){
		for(int j=0; j<ncol; j++){
			fscanf(f, "%f", &W[i*ncol+j]);
		}
	}
	fclose(f);
}

template<typename xpu>
void random_orthogonal(TensorContainer<xpu, 2> &node, float scale=1.0, float tauinv=1.0){
	int nrow = node.size(0);
	int ncol = node.size(1);
	float *W = (float *)malloc(nrow*ncol*sizeof(float));
	random_orthogonal_(W, nrow, ncol, tauinv);
	for(int i=0; i<nrow*ncol; i++){
//		cerr << W[i] << endl;
		W[i]*=scale;
	}
	TensorContainer<cpu, 2> node_cpu;
	node_cpu.Resize(node.shape_);
	for(int i=0; i<nrow; i++) for(int j=0; j<ncol; ++j) node_cpu[i][j] = W[i*ncol+j];
//	node_cpu.dptr=W;
//	node_cpu.shape[0]=ncol;
//	node_cpu.shape[1]=nrow;
//	node_cpu.stride_=ncol;
	Copy(node, node_cpu);
	free(W);
}



/*
void random_matrix(float *W, int output_dim, int input_dim, float spectral_radius){

	cerr << "ff random init start (" << input_dim << " * " << output_dim << ")" <<  endl;
	for(int i=0; i<output_dim; i++){
		for(int j=0; j<input_dim; j++){
			W[i*input_dim+j] = random(-1, 1);
		}
//		W_bias[i] = bias_scaling*random(-1, 1);
	}
	float *W_rec_tmp = (float *)malloc(output_dim*output_dim*sizeof(float));
//#pragma omp parallel for
	for(int i=0; i<output_dim; i++){
		for(int j=0; j<output_dim; j++){
			float tmp = 0;
			for(int k=0; k<input_dim; k++){
				tmp += W[i*input_dim+k]*W[j*input_dim+k];
			}
			W_rec_tmp[i*output_dim+j]=tmp;
		}
	}
//		cerr << "hoge" << endl;
	int ilo, ihi;
	float *eig_r = (float *)malloc(output_dim*sizeof(float));
	float *eig_i = (float *)malloc(output_dim*sizeof(float));
	float *vl = (float *)malloc(output_dim*sizeof(float));
	float *vr = (float *)malloc(output_dim*sizeof(float));
	float *scale = (float *)malloc(output_dim*sizeof(float));
	float *abnrm = (float *)malloc(output_dim*sizeof(float));
	float *rconde = (float *)malloc(output_dim*sizeof(float));
	float *rcondv = (float *)malloc(output_dim*sizeof(float));
//		for(int i=0; i<output_dim; i++){
//			for(int j=0; j<output_dim; j++){
//				cerr << W_rec_tmp[i*output_dim+j] << " ";
//						}
//						cerr << endl;
//		}
	LAPACKE_ssyevd(LAPACK_ROW_MAJOR, 'N', 'U', output_dim, W_rec_tmp, output_dim, eig_r);
//		LAPACKE_dgeevx(LAPACK_ROW_MAJOR, 'N', 'N', 'N', 'N', output_dim, W_rec_tmp, output_dim, eig_r, eig_i,
//			vl, output_dim, vr, output_dim, &ilo, &ihi, scale, abnrm, rconde, rcondv);

//		cerr << "piyo" << endl;
	float rad = 0;
	for(int i=0; i<output_dim; i++){
//			float tmp = sqrt(eig_r[i]*eig_r[i]+eig_i[i]*eig_i[i]);
		float tmp = sqrt(eig_r[i]*eig_r[i]);
		if(rad<tmp) rad = tmp;
	}
	cerr << "radius = " << rad << endl;
	rad = sqrt(rad);
	for(int i=0; i<input_dim*output_dim; i++){
		W[i] *= spectral_radius/rad;
	}

//		for(int i=0; i<output_dim; i++){
//			cerr << eig_r[i] << " " << eig_i[i] << endl;
//		}

	free(W_rec_tmp);
	free(eig_r);
	free(eig_i);
	free(vl);
	free(vr);
	free(scale);
	free(abnrm);
	free(rconde);
	free(rcondv);
	cerr << "ff random init done" << endl;
}
*/

#endif