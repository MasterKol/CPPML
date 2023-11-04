#include "dense.hpp"

#include <cassert>
#include <cmath>

#include "../random.hpp"
#include "../activation_func.hpp"
#include "../LinearAlgebra.hpp"

namespace CPPML {

bool Dense::compile_(){
	// this layer performs automatic flattening of inputs
	// add up input sizes of all input layers
	int input_nodes = 0;
	for(Layer* l : inputs){
		input_nodes += l->output_shape.size();
	}
	input_shape = Shape(input_nodes);
	
	num_weights = input_shape.size() * output_shape.size();
	num_biases = output_shape.size();

	num_params = num_weights + num_biases;
	intermediate_num = output_shape.size();

	return false;
}

void Dense::populate(float* params, float* gradients){
	weights = params + output_shape.size();
	biases = params;

	weight_grads = gradients + output_shape.size();
	bias_grads = gradients;

	// why set the weights between +-sqrt(6 / inputs) you may ask? I'm 
	// not sure but the internet said to do it so thats what I did....
	float r = sqrt(6.0f / (input_shape.size() + output_shape.size()));
	for(int i = 0; i < num_weights; i++){
		weights[i] = Random::randF(-r, r);
	}
	
	// biases are already zeroed and should stay that way
}

void Dense::compute(float* input, float* output, float* inter_ptr, bool training){
	if(inter_ptr == nullptr){
		inter_ptr = output;
	}

	// matrix multiply weights and input vector
	vDSP_mmul(weights, 1, input, 1, inter_ptr, 1, output_shape.size(), 1, input_shape.size());
	
	// add biases
	vDSP_vadd(inter_ptr, 1, biases, 1, inter_ptr, 1, output_shape.size());

	// apply activation function with intermediate as
	// input and output as output to move data if necessary
	if(activation)
		activation->f(inter_ptr, output, output_shape.size());
}

void Dense::get_change_grads(float* out_change, float* inpt_change, float* input, float* output, float* intermediate){
	// apply derivative of activation function to the values that
	// came out of this layer before they were passed through the
	// activation function

	// out_change <- activation'(intermediate) * out_change
	if(activation)
		activation->df(intermediate, out_change, output, out_change, output_shape.size());

	// calculate input change from output change
	// input_change^T <- out_change^T * weights
	vDSP_mmul(out_change, 1, weights, 1, inpt_change, 1, 1, input_shape.size(), output_shape.size());

	// claim gradient mutex so that gradients don't get trashed by
	// multiple threads accessing them at the same time
	// mutex expires when guard goes out of scope
	std::lock_guard<std::mutex> guard(gradient_mutex);

	// bias gradients: gradient of biases is just 1 * prev_change so just add prev_change to gradients
	// bias grad += intermediate
	vDSP_vadd(bias_grads, 1, out_change, 1, bias_grads, 1, output_shape.size());

	//weight gradients: grad matrix = grad matrix + prev_change * transpose(last_in)
	float* grad_row = weight_grads;
	float* prev_row = out_change;
	// loop over all rows in weight_grads and out_change
	for(int i = 0; i < output_shape.size(); i++){
		// add prev_change[i] * last_in to the i-th row of the weight grads
		vDSP_vsma(input, 1, prev_row, grad_row, 1, grad_row, 1, input_shape.size());

		// move to next rows
		grad_row += input_shape.size();
		prev_row++;
	}
}

} // namespace CPPML