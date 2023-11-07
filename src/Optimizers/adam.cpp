#include "adam.hpp"

#include <cstdlib>
#include <cmath>

#include "../LinearAlgebra.hpp"
#include "../shape.hpp"

namespace CPPML {

Adam::Adam(float learning_rate_, float learning_rate_falloff_, float beta1_, float beta2_, float epsilon_){
	beta1 = beta1_;
	beta2 = beta2_;
	beta1_hat = 1;
	beta2_hat = 1;
	epsilon = epsilon_;
	learning_rate = learning_rate_;
	learning_rate_falloff = learning_rate_falloff_;
	t = 0;
}

Adam::~Adam(){
	delete[] mt;
	delete[] vt;
}

void Adam::compile_(){
	mt = new float[net->num_params];
	memset(mt, 0, net->num_params * sizeof(float));
	vt = new float[net->num_params];
	memset(vt, 0, net->num_params * sizeof(float));
}

void Adam::update_params(){
	float* grads = net->gradients;
	float* params = net->params;
	const int num_params = net->num_params;

	// initial update
	t++;
	beta1_hat *= beta1;
	beta2_hat *= beta2;

	// compute mt <- grad * (1 - beta1) + mt * beta1
	vDSP_vintb(grads, 1, mt, 1, &beta1, mt, 1, num_params);
	
	// compute grad <- grad * grad
	vDSP_vsq(grads, 1, grads, 1, num_params);
	
	// compute vt <- grad * (1 - beta2) + vt * beta2
	vDSP_vintb(grads, 1, vt, 1, &beta2, vt, 1, num_params);

	// at this point gradient vector becomes a scratch variable

	// compute grad <- sqrt(vt)
	vvsqrtf(grads, vt, &num_params);

	// compute grad <- grad * sqrt(1 / (1 - beta2hat)) + epsilon
	const float sqrr1mh2h = sqrtf(1.0 / (1 - beta2_hat));
	vDSP_vsmsa(grads, 1, &sqrr1mh2h, &epsilon, grads, 1, num_params);

	// compute grad <- mt / grad
	vDSP_vdiv(grads, 1, mt, 1, grads, 1, num_params);

	// computes grad <- grad * lr / (beta1hat - 1)
	float lrd1mb1h = learning_rate / sqrtf(1 + t * learning_rate_falloff) / (beta1_hat - 1);
	vDSP_vsmul(grads, 1, &lrd1mb1h, grads, 1, num_params);

	// params += grads
	vDSP_vadd(grads, 1, params, 1, params, 1, num_params);
}

}