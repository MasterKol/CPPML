#include "activation.hpp"

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "LinearAlgebra.hpp"

namespace CPPML {

/**************** LINEAR ****************/
void linear_f(const float* input, float* output, int length){
	if(input != output){
		memcpy(output, input, length * sizeof(float));
	}
}

void linear_df(const float* input, float* output, int length){
	const float one = 1;
	vDSP_vfill(&one, output, 1, length);
}

/**************** ELU ****************/
void elu_f(const float* input, float* output, int length){
	float* t = new float[length];
	vDSP_vnabs(input, 1, t, 1, length); // t <- -abs(in)
	vvexpm1f(t, t, &length); // t <- e^t - 1
	vDSP_vmax(t, 1, input, 1, output, 1, length); // out <- max(t, in)
	delete[] t;
}

void elu_df(const float* input, float* output, int length){
	const float nInif = -__FLT_MAX__; // -infinity
	const float one = 1.0f; // one

	vvexpf(output, input, &length); // out <- e^in
	vDSP_vclip(output, 1, &nInif, &one, output, 1, length); // out <- min(out, 1)
}

/**************** RELU ****************/
void relu_f(const float* input, float* output, int length){
	const float zero = 0;

	vDSP_vthres(input, 1, &zero, output, 1, length); // out <- max(out, 0)
}

void relu_df(const float* input, float* output, int length){
	const float zero = 0;
	const float one = 1;

	vDSP_vthrsc(input, 1, &zero, &one, output, 1, length); // out <- sign(in)
	vDSP_vthres(output, 1, &zero, output, 1, length); // out <- max(out, 0)
}

/**************** SIGMOID ****************/
void sigmoid_f(const float* input, float* output, int length){ // computes 1 / (1 + e^-x) for all x in d
	vDSP_vneg(input, 1, output, 1, length); // out <- -in
	vvexpf(output, output, &length); // out <- e^out
	const float one = 1.0;
	vDSP_vsadd(output, 1, &one, output, 1, length); // out <- out + 1
	vvrecf(output, output, &length); // out <- 1/out
}

void sigmoid_df(const float* input, float* output, int length){ // computes (1 / (1 + e^-x)), x <- t * t - t for all x in d
	vDSP_vneg(input, 1, output, 1, length); // out <- -in
	vvexpf(output, output, &length); // out <- e^out
	const float one = 1.0;
	vDSP_vsadd(output, 1, &one, output, 1, length); // out <- out + 1
	vvrecf(output, output, &length); // out <- 1/out

	vDSP_vmsb(output, 1, output, 1, output, 1, output, 1, length); // out <- out * out + out
	vDSP_vneg(output, 1, output, 1, length); // out <- -out
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

void softmax_df(const float* input, float* output, int length){
	float dt;
	// dt = out . in
	vDSP_dotpr(output, 1, input, 1, &dt, length);
	
	// out -= dt
	dt = -dt;
	vDSP_vsadd(input, 1, &dt, output, 1, length);
	
	// out = in .* out
	vDSP_vmul(input, 1, output, 1, output, 1, length);
}

}