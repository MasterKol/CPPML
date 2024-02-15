#include "group_norm.hpp"

#include <iostream>
#include <cmath>
#include <mutex>

#include "../shape.hpp"
#include "../layer.hpp"
#include "../LinearAlgebra.hpp"
#include "../random.hpp"

namespace CPPML {

const float epsilon = 1e-10;

bool GroupNorm::compile_(){
	input_shape = inputs[0]->output_shape;
	input_shape.d(0); // set to zero because it will be re added

	// size of one input slice, every input must be a multiple of this
	const int multiple = input_shape.w() * input_shape.h();

	// loop over inputs and generate input shape
	for(Layer* l : inputs){
		Shape os = l->output_shape;
		// check if inputs match in the correct dimensions
		// if the input is flat try to fix it else throw error
		if(!((os.d() == 1 && os.h() == 1 && os.w() % multiple == 0) ||
		   (os.h() != 1 && os.w() == input_shape.w() && os.h() == input_shape.h()))){
			std::cerr << "CNN Dimentions do not match.\n\tExpected: (" << input_shape.w() << ", "
				<< input_shape.w() << ") got: (" << os.w() << ", " << os.h() << ")\n";
			exit(-1);
		}
		input_shape.d(input_shape.d() + os.size() / multiple);
	}

	output_shape = input_shape;
	num_params = 2 * num_groups;

	intermediate_num = num_groups;

	return false;
}

void GroupNorm::populate(float* params_, float* gradients_){
	// gammas = params;
	// gamma_grads = gradients;

	// betas = params + num_groups;
	// beta_grads = gradients + num_groups;

	params = params_;
	gradients = gradients_;

	// initialization
	for(int i = 0; i < num_groups * 2; i+=2){ // beta
		params[i] = 0;
	}

	for(int i = 1; i < num_groups * 2; i+=2){ // gammma
		params[i] = 1;
	}
}

void norm_group(float* input, float* output, float* denom, float beta, float gamma, int size){
	float mean = 0;
	vDSP_sve(input, 1, &mean, size);
	mean /= size;

	*denom = 0;
	vDSP_dotpr(input, 1, input, 1, denom, size); // use dotpr and not vDSP_svesq because vDSP_svesq given random outputs
	*denom = *denom / size - mean * mean; // var = E[x^2] - (E[x])^2
	*denom = 1.0f / std::sqrt(*denom + epsilon);

	float factor = gamma * (*denom);
	float add = beta - mean * factor;

	vDSP_vsmsa(input, 1, &factor, &add, output, 1, size);
	/*float stdev;
	vDSP_normalize(input, 1, output, 1, mean, &stdev, size);
	
	if(stdev == 0) // if stdev is 0 then outputs will be nan, instead replace them with 0
		memset(output, 0, size * sizeof(float));
	*var = stdev * stdev;

	vDSP_vsmsa(output, 1, &gamma, &beta, output, 1, size);*/
}

void GroupNorm::compute(float* input, float* output, float* intermediate_buffer, bool training){
	const int slice = input_shape.w() * input_shape.h();
	const int group_size = input_shape.d() / num_groups * slice;
	// const int over_hang_size = input_shape.d() * slice - group_size * num_groups;
	memset(output, 0, output_shape.size() * sizeof(float));

	float t_denom = 0;
	float* denom = &t_denom;
	if(intermediate_buffer)
		denom = intermediate_buffer;

	float* beta = params;
	float* gamma = params + 1;

	int offset = 0;
	for(int g = 0; g < num_groups; g++){
		norm_group(input + offset, output + offset, denom, *beta, *gamma, group_size);
		offset += group_size;

		beta += 2;
		gamma += 2;
		if(intermediate_buffer)
			denom++;
	}

	// norm_group(input + offset, output + offset, denom, *beta, *gamma, over_hang_size);
}

void group_grad(float* out_change, float* in_change, float* output, float beta, float gamma, float* beta_grad, float* gamma_grad, float inv_var, int size){
	// beta grad is sum of output change
	*beta_grad = 0;
	vDSP_sve(out_change, 1, beta_grad, size);

	// store temp dot product out out_change and output in gamma_grad
	*gamma_grad = 0;
	vDSP_dotpr(out_change, 1, output, 1, gamma_grad, size);

	// std::cout << *gamma_grad << ", ";

	*gamma_grad = (*gamma_grad - beta * (*beta_grad)) / gamma;
	// std::cout << beta << ", " << *beta_grad << ", " << gamma << ", " << *gamma_grad << std::endl;

	const float denom = inv_var / size;
	
	const float const_add = denom * (beta * (*gamma_grad) - gamma * (*beta_grad));
	const float out_grad_scale = gamma * inv_var;
	const float out_scale = -denom * (*gamma_grad);

	// std::cout << "   " << const_add << ", " << out_grad_scale << ", " << out_scale << std::endl;
	
	// int o = 0;
	// for(int i = 0; i < size / 3; i++){
	// 	std::cout << out_change[o++] << ", " << out_change[o++] << ", " << out_change[o++] << std::endl;
	// }
	// std::cout << std::endl;

	// o = 0;
	// for(int i = 0; i < size / 3; i++){
	// 	std::cout << output[o++] << ", " << output[o++] << ", " << output[o++] << std::endl;
	// }
	// std::cout << std::endl;

	// in_change = out_change * out_grad_scale + output * out_scale
	vDSP_vsmsma(out_change, 1, &out_grad_scale, output, 1, &out_scale, in_change, 1, size);

	// in_change += const_add
	vDSP_vsadd(in_change, 1, &const_add, in_change, 1, size);
	// memset(in_change, 1, size * sizeof(float));
}

void GroupNorm::get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate){
	//
	const int slice = input_shape.w() * input_shape.h();
	const int group_size = input_shape.d() / num_groups * slice;
	// std::cout << group_size << std::endl;
	// const int over_hang_size = input_shape.d() * slice - group_size * num_groups;
	
	float t_grads[num_groups * 2];

	float* beta = params;
	float* gamma = params + 1;

	float* beta_grad = &t_grads[0];
	float* gamma_grad = (&t_grads[1]);

	int offset = 0;
	for(int g = 0; g < num_groups; g++){
		group_grad(out_change + offset, inpt_change + offset, output + offset, *beta, *gamma, beta_grad, gamma_grad, intermediate[g], group_size);
		
		offset += group_size;
		beta += 2;
		gamma += 2;
		beta_grad += 2;
		gamma_grad += 2;
	}

	// group_grad(out_change + offset, inpt_change + offset, output + offset, *beta, *gamma, beta_grad, gamma_grad, intermediate[num_groups-1], over_hang_size);

	std::lock_guard<std::mutex> guard(gradient_mutex);

	vDSP_vadd(gradients, 1, t_grads, 1, gradients, 1, num_groups * 2);
}

} // namespace CPPML