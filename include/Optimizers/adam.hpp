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
	/// @brief Adam optimizer, most params should be left at default unless you
	///		   know what you are doing.
	/// @param learning_rate 
	/// @param beta1 
	/// @param beta2 
	/// @param epsilon 
	Adam(float learning_rate=0.001f, float beta1=0.9f,
			float beta2=0.999f, float epsilon=1E-7);
	~Adam();
	virtual void update_params();
private:
	virtual void compile_();
};

}

#endif