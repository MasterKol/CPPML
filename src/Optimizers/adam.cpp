#include "adam.hpp"

#include "../LinearAlgebra.hpp"
#include <stdlib.h>
#include <math.h>

#include "../data.hpp"

namespace CPPML {

Adam::Adam(float learning_rate_, float beta1_, float beta2_, float epsilon_){
	beta1 = beta1_;
	beta2 = beta2_;
	beta1_hat = 1;
	beta2_hat = 1;
	epsilon = epsilon_;
	learning_rate = learning_rate_;
	t = 0;
}

void Adam::compile_(){
	mt = new float[net->num_params];
	memset(mt, 0, net->num_params * sizeof(float));
	vt = new float[net->num_params];
	memset(vt, 0, net->num_params * sizeof(float));
}

void Adam::update_params(){
	float* __restrict grads = net->gradients;
	float* __restrict params = net->params;
	const int num_params = net->num_params;

	// initial update
	t++;
	beta1_hat *= beta1;
	beta2_hat *= beta2;

	// const float b1m1 = 1 - beta1;
	// const float b2m1 = 1 - beta2;
	const float sqrr1mh2h = sqrtf(1.0 / (1 - beta2_hat));
	const float lrd1mb1h = learning_rate / (beta1_hat - 1);
	// const float b1 = beta1;
	// const float b2 = beta2;

	// float* __restrict mt_ = mt;
	// float* __restrict vt_ = vt;

	/*#pragma omp parallel for
	for(int i = 0; i < num_params; i++){
		mt_[i] = grads[i] * b1m1 + mt_[i] * b1;
		grads[i] = grads[i] * grads[i];
		vt_[i] = grads[i] * b2m1 + vt_[i] * b2;
		params[i] += mt_[i] * lrd1mb1h / (sqrtf(vt_[i]) * sqrr1mh2h + epsilon);
		//grads[i] = mt[i] / (sqrtf(vt[i]) * sqrr1mh2h + epsilon);
		//grads[i] = sqrt(vt[i]);
		//grads[i] = grads[i] * sqrr1mh2h + epsilon;
		//grads[i] = mt[i] / grads[i];
		//params[i] += grads[i] * learning_rate / (beta1_hat - 1);
		//grads[i] *= lrd1mb1h;
		//params[i] += grads[i];
	}*/

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
	//const float sqrr1mh2h = sqrtf(1.0 / (1 - beta2_hat));
	vDSP_vsmsa(grads, 1, &sqrr1mh2h, &epsilon, grads, 1, num_params);

	// compute grad <- mt / grad
	vDSP_vdiv(grads, 1, mt, 1, grads, 1, num_params);

	// computes grad <- grad * lr / (beta1hat - 1)
	//float lrd1mb1h = learning_rate / (beta1_hat - 1);
	vDSP_vsmul(grads, 1, &lrd1mb1h, grads, 1, num_params);

	// params += grads
	vDSP_vadd(grads, 1, params, 1, params, 1, num_params);
}

}