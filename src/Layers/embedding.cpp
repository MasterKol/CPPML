#include "embedding.hpp"

#include <iostream>
#include <cmath>
#include <mutex>

#include "../layer.hpp"
#include "../shape.hpp"
#include "../random.hpp"
#include "../LinearAlgebra.hpp"

namespace CPPML {

Embedding::Embedding(int num_classes, int embedding_length, Layer* input_layer) : 
	num_classes(num_classes), embedding_length(embedding_length){
	if(num_classes < 1){
		std::cerr << "num_classes must be >= 1\n";
		exit(-1);
	}
	if(embedding_length < 1){
		std::cerr << "embedding_length must be >= 1\n";
		exit(-1);
	}
	if(input_layer)
		add_input(input_layer);
}

bool Embedding::compile_(){
	if(inputs.size() != 1){
		std::cerr << "Embedding layer may only have exactly 1 input layer\n";
		exit(-1);
	}

	if(inputs[0]->output_shape.size() != 1){
		std::cerr << "Embedding input layer outputshape must have a size of 1\n";
	}

	num_params = num_classes * embedding_length;

	input_shape = 1;
	output_shape = embedding_length;
	intermediate_num = 0;

	return false;
}

void Embedding::populate(float* params_, float* gradients_){
	params = params_;
	gradients = gradients_;

	Random::fillGaussian(params_, num_params, 0, 1);
}

void Embedding::compute(float* input, float* output, float* intermediate_buffer, bool training){
	int index = std::round(input[0]);

	if(0 <= index && index < num_classes){ // inside range, copy embedding to output
		memcpy(output, params + embedding_length * index, embedding_length * sizeof(float));
	}else{ // outsize range, set embedding to 0
		memset(output, 0, embedding_length * sizeof(float));
	}
}

void Embedding::get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate){
	//
	int index = std::round(input[0]);
	inpt_change[0] = 0;

	if(index < 0 || index >= num_classes) // outside range, no gradients, just return
		return;

	float* grad_pos = gradients + index * embedding_length;

	const std::lock_guard<std::mutex> lock(gradient_mutex);

	vDSP_vadd(grad_pos, 1, out_change, 1, grad_pos, 1, embedding_length);
}

} // namespace CPPML 