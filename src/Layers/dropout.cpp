#include "dropout.hpp"

#include <iostream>
#include <limits>
#include <mutex>

#include "../shape.hpp"
#include "../random.hpp"
#include "../LinearAlgebra.hpp"

namespace CPPML {

Dropout::Dropout(double dropout_ratio, Layer* input_layer) : dropout_ratio(dropout_ratio) {
	if(input_layer != nullptr)
		add_input(input_layer);
}

bool Dropout::compile_(){
	if(inputs.size() != 1){
		std::cerr << "Dropout layers only accept 1 input layer\n";
		exit(-1);
	}

	input_shape = inputs[0]->output_shape;
	output_shape = inputs[0]->output_shape;

	num_params = 0;
	intermediate_num = 1;

	return false;
}

bool random_prob(unsigned int& rng_state, unsigned int cutoff){
    // Xorshift algorithm from George Marsaglia's paper
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state < cutoff;
}

void Dropout::compute(float* input, float* output, float* intermediate_buffer, bool training){
	if(!training){
		float p = 1 - dropout_ratio;
		vDSP_vsmul(input, 1, &p, output, 1, input_shape.size());
		return;
	}

	// copy inputs to outputs
	memcpy(output, input, input_shape.size() * sizeof(float));

	// get a random number from global rng in a threadsafe way for use as a seed
	rng_mutex.lock();
	std::uniform_int_distribution<int> distribution (0, std::numeric_limits<int>::max());
	unsigned int rng_state = distribution(Random::rng);
	rng_mutex.unlock();
	((unsigned int*)intermediate_buffer)[0] = rng_state;

	unsigned int cutoff = (unsigned int)(std::numeric_limits<unsigned int>::max() * dropout_ratio);
	for(int i = 0; i < input_shape.size(); i++){
		if(!random_prob(rng_state, cutoff))
			continue;
		output[i] = 0;
	}
}

void Dropout::get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate){
	//
	unsigned int rng_state = ((unsigned int*)intermediate)[0];

	memcpy(inpt_change, out_change, input_shape.size());
	unsigned int cutoff = (unsigned int)(std::numeric_limits<unsigned int>::max() * dropout_ratio);
	for(int i = 0; i < input_shape.size(); i++){
		if(!random_prob(rng_state, cutoff))
			continue;
		inpt_change[i] = 0;
	}
}

} // namespace CPPML