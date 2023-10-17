#include <cstring>
#include <cmath>
#include <iostream>

#include "random.hpp"
#include "network.hpp"
#include "layer.hpp"
#include "Layers/input.hpp"
#include "cost_func.hpp"

int input_length, output_length;

float h = 1 / 256.0f;
float epsilon = 1e-2;

float *input, *target, *gradients, *input_change;
CPPML::Network* net;

float original_loss = 0;

float getDerv(float*);
void setup(int);

/// @brief Checks derivative of network inputs and compares
///		   them to empirical value
void checkInputGradients(){
	float avg = 0;
	for(int i = 0; i < input_length; i++){
		float calc = getDerv(input + i);
		if(calc == 0)
			continue;
		float err = std::abs((calc - input_change[i]) / calc);
		avg += err;

		//std::cerr << i << ", " << err << ", " << calc << ", " << input_change[i] << "\n";
	}
	avg /= input_length;

	std::cerr << "Input derivative avg error = " << avg * 100 << "%\n";
	if(avg > epsilon || avg < 0 || isnan(avg)){
		exit(-1);
	}
}

/// @brief Checks derivative of network parameters and compares
// 		   them to empirical value
void checkParameterGradients(){
	if(net->num_params == 0)
		return;  // some layers don't have parameters so just exit early
	
	float avg = 0;
	for(int i = 0; i < net->num_params; i++){
		float calc = getDerv(net->params + i);
		if(calc == 0)
			continue;
		float err = std::abs((calc - gradients[i]) / calc);
		avg += err;

		//std::cerr << i << ", " << err << ", " << calc << ", " << gradients[i] << "\n";
	}
	avg /= net->num_params;

	std::cerr << "Parameter derivative average error = " << avg * 100 << "%\n";
	if(avg > epsilon || avg < 0 || isnan(avg)){
		exit(-1);
	}
}

/// @brief evaluates network with given value changed by some amount
/// @param toChange pointer to value to change
/// @param dx amount to change value by
/// @return cost of the network with given value changed
float eval(float* toChange, float dx){
	float t = *toChange;
	(*toChange) += dx;
	float out = 0;
	net->fit_network(input, target, 1, &out);
	*toChange = t;
	return out;
}

/// @brief Gets numerical derivative of given value in the network
/// @param toChange Pointer to the value to be changed
/// @return Derivative cost wrt given value
float getDerv(float* toChange){
	float costmh  = -8 * eval(toChange, -  h);
	float costph  =  8 * eval(toChange,    h);
	float cost2mh =      eval(toChange, -2*h);
	float cost2ph = -    eval(toChange,  2*h);

	return (costph + costmh + cost2ph + cost2mh) / (12 * h);
}

/// @brief Sets up net for a single layer to be tested
/// @param layer layer to be tested, will be added to a network
/// @param input_shape shape of input to the network (and the given layer)
/// @param seed seed for random number generator (random if 0)
void setup(CPPML::Layer* layer, CPPML::Shape input_shape, int seed = 0){
	net = new CPPML::Network(CPPML::MAE);

	CPPML::Layer* l = new CPPML::Input(input_shape, net);
	layer->add_input(l);
	setup(seed);
}

/// @brief allows user to setup net, compiles and does all setup afterwards
/// @param seed seed for random number generator (random if 0)
void setup(int seed = 0){
	/**** SETUP ****/
	if(seed == 0){
		std::cerr << "random seed: " << CPPML::Random::time_seed() << "\n";
	}else{
		CPPML::Random::rand_seed(seed);
		std::cerr << "random seed: " << seed << "\n";
	}
	
	// make network
	net->compile(nullptr);

	output_length = net->output_length;
	input_length = net->input_length;

	// initialize memory
	input = new float[input_length];
	target = new float[output_length];
	gradients = new float[net->num_params];
	input_change = new float[net->last_io_size];

	// fill input with random values
	CPPML::Random::fillGaussian(input, input_length, 0, 1);

	// fill output with random values
	float* output = new float[output_length];
	net->eval(input, output);
	for(int i = 0; i < output_length; i++){
		target[i] = output[i] + 10 * h;
	}

	// get original loss, input change, and weight gradients
	net->fit_network(input, target, nullptr, nullptr, input_change, &original_loss);
	memcpy(gradients, net->gradients, net->num_params * sizeof(float));
}