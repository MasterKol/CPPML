#include "activation.hpp"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "LinearAlgebra.hpp"

namespace CPPML {

/**************** LINEAR ****************/
void linear_f(float* input, float* output, int length){
	if(input != output){
		memcpy(output, input, length * sizeof(float));
	}
}

void linear_df(float* input, float* output, int length){
	const float one = 1;
	vDSP_vfill(&one, output, 1, length);
}

/**************** ELU ****************/
void elu_f(float* input, float* output, int length){
	float* t = new float[length];
	vDSP_vnabs(input, 1, t, 1, length); // t <- -abs(in)
	vvexpm1f(t, t, &length); // t <- e^t - 1
	vDSP_vmax(t, 1, input, 1, output, 1, length); // out <- max(t, in)
	delete[] t;
}

void elu_df(float* input, float* output, int length){
	const float nInif = -__FLT_MAX__; // -infinity
	const float one = 1.0f; // one

	vvexpf(output, input, &length); // out <- e^in
	vDSP_vclip(output, 1, &nInif, &one, output, 1, length); // out <- min(out, 1)
}

/**************** RELU ****************/
void relu_f(float* input, float* output, int length){
	const float zero = 0;

	vDSP_vthres(input, 1, &zero, output, 1, length); // out <- max(out, 0)
}

void relu_df(float* input, float* output, int length){
	const float zero = 0;
	const float one = 1;

	vDSP_vthrsc(input, 1, &zero, &one, output, 1, length); // out <- sign(in)
	vDSP_vthres(output, 1, &zero, output, 1, length); // out <- max(out, 0)
}

/**************** SIGMOID ****************/
void sigmoid_f(float* input, float* output, int length){ // computes 1 / (1 + e^-x) for all x in d
	vDSP_vneg(input, 1, output, 1, length); // out <- -in
	vvexpf(output, output, &length); // out <- e^out
	const float one = 1.0;
	vDSP_vsadd(output, 1, &one, output, 1, length); // out <- out + 1
	vvrecf(output, output, &length); // out <- 1/out
}

void sigmoid_df(float* input, float* output, int length){ // computes (1 / (1 + e^-x)), x <- t * t - t for all x in d
	vDSP_vneg(input, 1, output, 1, length); // out <- -in
	vvexpf(output, output, &length); // out <- e^out
	const float one = 1.0;
	vDSP_vsadd(output, 1, &one, output, 1, length); // out <- out + 1
	vvrecf(output, output, &length); // out <- 1/out

	vDSP_vmsb(output, 1, output, 1, output, 1, output, 1, length); // out <- out * out + out
	vDSP_vneg(output, 1, output, 1, length); // out <- -out
}

}