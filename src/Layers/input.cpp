#include "input.hpp"

#include "../layer.hpp"

Input::Input(Shape input_shape_, Network* net){
	output_shape = input_shape_;

	if(net != NULL){
		net->add_input_layer(this);
	}
}

bool Input::compile_(){
	return true;
}

void Input::populate(float* params, float* gradients){
	return; // nothing to do
}

void Input::compute(float* input, float* output, float* intermediate_buffer){
	return; // nothing to do
}

void Input::get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate){
	return; // nothing to do (input doesn't need to propigate gradients)
}