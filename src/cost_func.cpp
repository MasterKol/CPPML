#include "cost_func.hpp"

#include <cmath>
#include <memory>

#include "LinearAlgebra.hpp"

namespace CPPML {

const float epsilon = 1e-10;

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
	std::unique_ptr<float[]> t(new float[length]);
	vDSP_vsub(y, 1, x, 1, t.get(), 1, length); // t = x - y
	vvfabsf(t.get(), t.get(), &length); // t = |t|

	float absum = 0;
	vDSP_sve(t.get(), 1, &absum, length); // absum = sum(t)
	return absum / (float)length;
}

void mae_get_cost_derv(float* x, float* y, float* out, int length){
	// calculate out <- (x - y) / x.length
	float rlength = 1.0f / length;
	vDSP_vfill(&rlength, out, 1, length); // fill output array with magnitude of final output

	std::unique_ptr<float[]> t(new float[length]);
	vDSP_vsub(y, 1, x, 1, t.get(), 1, length); // calculate t <- x - y
	vvcopysignf(out, out, t.get(), &length); // copy sign of x - y to output array
}

/**************** Huber ****************/
float huber_get_cost(float* x, float* y, int length){
	float out = 0;
	float diff;
	for(int i = 0; i < length; i++){
		diff = std::abs(x[i] - y[i]);
		out += (diff < 1) ? diff * diff * 0.5 : diff - 0.5;
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
		if(x[i] < 0 || y[i] <= 0)
			continue;
		out -= y[i] * log(x[i] + epsilon);
	}
	return out;
}

void cross_entropy_get_cost_derv(float* x, float* y, float* out, int length){
	vDSP_vsadd(x, 1, &epsilon, out, 1, length); // out = x + epsilon
	vDSP_vdiv(out, 1, y, 1, out, 1, length); // out = y / out
	vDSP_vneg(out, 1, out, 1, length); // out = -out
}

}