#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "../optimizer.hpp"
#include "../network.hpp"

namespace CPPML {

/*
 * Implements the adam optimizer from this paper
 * https://arxiv.org/abs/1412.6980
 */
class Adam : public Optimizer {
private:
	float beta1, beta2;
	float beta1_hat, beta2_hat;
	float epsilon;
	float* mt;
	float* vt;
	int t;
public:
	float learning_rate;
	
	Adam(float learning_rate_=0.001f, float beta1_=0.9f,
			float beta2_=0.999f, float epsilon_=1E-7);
	virtual void update_params();
	virtual void compile_();
};

}

#endif