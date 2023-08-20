#include "sgd.hpp"
#include "../LinearAlgebra.hpp"

namespace CPPML {

SGD::SGD(float learning_rate_){
	learning_rate = learning_rate_;
}

void SGD::compile_(){}

void SGD::update_params(){
	float* grads = net->gradients;
	float* params = net->params;
	const int num_params = net->num_params;

	// grads *= learning_rate
	vDSP_vsmul(grads, 1, &learning_rate, grads, 1, num_params);

	// params += grads
	vDSP_vsub(grads, 1, params, 1, params, 1, num_params);
}

}