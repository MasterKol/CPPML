#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "../optimizer.hpp"
#include "../network.hpp"

namespace CPPML {

class Adam : public Optimizer {
public:
	float beta1, beta2;
	float beta1_hat, beta2_hat;
	float epsilon;
	float learning_rate;
	float* mt;
	float* vt;
	int t;
	Adam(float learning_rate_=0.001f, float beta1_=0.9f,
			float beta2_=0.999f, float epsilon_=1E-7);
	virtual void update_params();
	virtual void compile_();
};

}

#endif