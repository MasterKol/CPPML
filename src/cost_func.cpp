#include "cost_func.hpp"

#include <cmath>

#include "LinearAlgebra.hpp"

namespace CPPML {

/**************** Mean Squared Error ****************/
float mse_get_cost(float* x, float* y, int length){
	// calculate sum((x - y)^2) / x.length
	float dsq = 0;
	vDSP_distancesq(x, 1, y, 1, &dsq, length);
	return 0.5f * dsq / length;
}

void mse_get_cost_derv(float* x, float* y, float* out, int length){
	// calculate out <- (x - y) / x.length
	float rlength = 1.0f / length;
	vDSP_vsbsm(x, 1, y, 1, &rlength, out, 1, length);
}

/**************** Mean Absolute Error ****************/
float mae_get_cost(float* x, float* y, int length){
	vDSP_vsub(y, 1, x, 1, x, 1, length);
	float absum;
	vDSP_svemg(x, 1, &absum, length);
	//float absum = cblas_sasum(length, x, 1);
	vDSP_vadd(y, 1, x, 1, x, 1, length);

	return absum / length;
}

void mae_get_cost_derv(float* x, float* y, float* out, int length){
	// calculate out <- (x - y) / x.length
	float rlength = 1.0f / length;
	vDSP_vfill(&rlength, out, 1, length); // fill output array with magnitude of final output

	vDSP_vsub(y, 1, x, 1, x, 1, length); // calculate x <- x - y
	vvcopysignf(out, out, x, &length); // copy sign of x - y to output array
	vDSP_vadd(y, 1, x, 1, x, 1, length); // reset x by calculating x <- x + y
}

/**************** Huber ****************/
float huber_get_cost(float* x, float* y, int length){
	float out = 0;
	float diff;
	for(int i = 0; i < length; i++){
		diff = fabs(x[i] - y[i]);
		out += (diff < 1) ? diff * diff : diff;
	}

	return out / length;
}

void huber_get_cost_derv(float* x, float* y, float* out, int length){
	// calculate out <- (x - y) / x.length
	
	//vDSP_vfill(&rlength, out, 1, x->length); // fill output array with magnitude of final output

	vDSP_vsub(y, 1, x, 1, out, 1, length); // calculate out <- x - y
	float one = 1;
	float none = -1;
	vDSP_vclip(out, 1, &none, &one, out, 1, length); // clip out to -1, 1

	float rlength = 1.0f / length;
	vDSP_vsmul(out, 1, &rlength, out, 1, length); // multiply out by 1/length
}

/**************** Cross Entropy ****************/
float cross_entropy_get_cost(float* x, float* y, int length){
	float out = 0;
	for(int i = 0; i < length; i++){
		if(y[i] == 0)
			continue;
		out -= y[i] * log(x[i]);
	}
	return out;
}

void cross_entropy_get_cost_derv(float* x, float* y, float* out, int length){
	vDSP_vsub(y, 1, x, 1, out, 1, length);
}

}