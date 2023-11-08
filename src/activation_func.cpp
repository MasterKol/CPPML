#include "activation_func.hpp"

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "shape.hpp"
#include "LinearAlgebra.hpp"

namespace CPPML {

const float epsilon = 1e-10;

/**************** LINEAR ****************/
void linear_f(const float* input, float* output, int length){
	if(input != output){
		memcpy(output, input, length * sizeof(float));
	}
}

void linear_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length){
	//const float one = 1;
	//vDSP_vfill(&one, output, 1, length);
	if(input_gradients != output_gradients){
		memcpy(input_gradients, output_gradients, length * sizeof(float));
	}
}

/**************** ELU ****************/
void elu_f(const float* input, float* output, int length){
	float* t = new float[length];
	vDSP_vnabs(input, 1, t, 1, length); // t <- -abs(in)
	vvexpm1f(t, t, &length); // t <- e^t - 1
	vDSP_vmax(t, 1, input, 1, output, 1, length); // out <- max(t, in)
	delete[] t;
}

void elu_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length){
	const float nInif = -__FLT_MAX__; // -infinity
	const float one = 1.0f; // one

	vvexpf(output, input, &length); // out <- e^in
	vDSP_vclip(output, 1, &nInif, &one, output, 1, length); // out <- min(out, 1)

	// in_grad = out_grad * Jacobian (out)
	vDSP_vmul(output, 1, output_gradients, 1, input_gradients, 1, length);
}

/**************** RELU ****************/
void relu_f(const float* input, float* output, int length){
	const float zero = 0;

	vDSP_vthres(input, 1, &zero, output, 1, length); // out <- max(out, 0)
}

void relu_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length){
	const float zero = 0;
	const float one = 1;

	vDSP_vthrsc(input, 1, &zero, &one, output, 1, length); // out <- sign(in)
	vDSP_vthres(output, 1, &zero, output, 1, length); // out <- max(out, 0)

	// in_grad = out_grad * Jacobian (out)
	vDSP_vmul(output, 1, output_gradients, 1, input_gradients, 1, length);
}

/**************** SIGMOID ****************/
void sigmoid_f(const float* input, float* output, int length){ // computes 1 / (1 + e^-x) for all x in d
	vDSP_vneg(input, 1, output, 1, length); // out <- -in
	vvexpf(output, output, &length); // out <- e^out
	const float one = 1.0;
	vDSP_vsadd(output, 1, &one, output, 1, length); // out <- out + 1
	vvrecf(output, output, &length); // out <- 1/out
}

void sigmoid_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length){ // computes (1 / (1 + e^-x)), x <- t * t - t for all x in d
	// out_grads *= output
	vDSP_vmul(output, 1, output_gradients, 1, output_gradients, 1, length);

	// output = -output
	vDSP_vneg(output, 1, output, 1, length);

	// output += 1
	const float one = 1;
	vDSP_vsadd(output, 1, &one, output, 1, length);

	// out_grads *= output
	vDSP_vmul(output, 1, output_gradients, 1, input_gradients, 1, length);
}

/**************** TANH ****************/
void tanh_f(const float* input, float* output, int length){
	vvtanhf(output, input, &length);
}

void tanh_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length){
	// 1 - out^2

	// output <- output * output
	vDSP_vmul(output, 1, output, 1, output, 1, length);
	
	// output <- -output
	vDSP_vneg(output, 1, output, 1, length);

	// output += 1
	const float one = 1;
	vDSP_vsadd(output, 1, &one, output, 1, length);

	// in_grad = out_grad * Jacobian (out)
	vDSP_vmul(output, 1, output_gradients, 1, input_gradients, 1, length);
}

/**************** SOFTMAX ****************/
void softmax_f(const float* input, float* output, int length){
	// this slightly odd implementation of softmax gives more
	// numerical stability than direct application of the formula

	float max;
	// find the maximum value in v
	vDSP_maxv(input, 1, &max, length);

	// subtract max off of v
	max = -max;
	vDSP_vsadd(input, 1, &max, output, 1, length);

	// exp v
	vvexpf(output, output, &length);

	// sum v
	float total;
	vDSP_sve(output, 1, &total, length);

	// divide v by total
	vDSP_vsdiv(output, 1, &total, output, 1, length);
}

// switch to using output not input for grad
void softmax_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length){
	float dt; // dot product of output and gradient of outputs
	vDSP_dotpr(output_gradients, 1, output, 1, &dt, length); // calc. dot product
	dt = -dt; // negate dot product
	
	// calculate (out_grads + dt) * (output + epsilon) -> input_grads
	vDSP_vaam(output_gradients, 1, &dt, 0, output, 1, &epsilon, 0, input_gradients, 1, length);
}

}