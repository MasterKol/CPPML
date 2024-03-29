#include "activation.hpp"
#include <iostream>

namespace CPPML {

bool ActivationLayer::compile_(){
	if(inputs.size() == 1){
		input_shape = inputs[0]->output_shape;
		output_shape = input_shape;
	}else{
		int len = 0;
		for(Layer* l : inputs){
			len += l->output_shape.size();
		}
		
		input_shape = Shape(len);
		output_shape = Shape(len);
	}
	intermediate_num = 0;
	return false;
}

void ActivationLayer::compute(float* input, float* output, float* intermediate_buffer, bool training){
	act->f(input, output, input_shape.size());
}

void ActivationLayer::get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate){
	act->df(input, inpt_change, output, out_change, input_shape.size());
}

} // namespace CPPML