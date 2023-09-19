#include <cstring>
#include <cmath>
#include <iostream>

#include "../include/random.hpp"
#include "../include/network.hpp"
#include "../include/layer.hpp"
#include "../include/Layers/input.hpp"
#include "../include/cost_func.hpp"

extern CPPML::Shape input_shape;
CPPML::Shape output_shape;

const float h = 1e-4;
const float epsilon = 1e-2;

float *input, *target, *gradients, *input_change;
CPPML::Network* net;

float original_loss = 0;

void setup(CPPML::Layer* layer, int seed = 0){
	/**** SETUP ****/
	if(seed == 0)
		std::cerr << "random seed: " << CPPML::Random::time_seed() << "\n";
	
	// make network
	net = new CPPML::Network(CPPML::MAE);

	CPPML::Layer* l = new CPPML::Input(input_shape, net);
	layer->add_input(l);

	net->compile(nullptr);

	output_shape = net->output_layer->output_shape;

	// initialize memory
	input = new float[input_shape.size()];
	target = new float[output_shape.size()];
	gradients = new float[net->num_params];
	input_change = new float[net->last_io_size];

	// fill input with random values
	CPPML::Random::fillGaussian(input, input_shape.size(), 0, 1);
	/*input[0] = CPPML::Random::randF(-0.1, 0.1);
	for(int i = 1; i < input_shape.size(); i++){
		input[i] = (std::abs(input[i - 1]) + CPPML::Random::randF(3, 10) * h) * (CPPML::Random::randI(1) * 2 - 1);
	}*/

	// fill output with random values
	float* output = new float[output_shape.size()];
	net->eval(input, output);
	for(int i = 0; i < output_shape.size(); i++){
		target[i] = output[i] + 10 * h;
	}

	// get original loss, input change, and weight gradients
	net->fit_network(input, target, nullptr, nullptr, input_change, &original_loss);
	memcpy(gradients, net->gradients, net->num_params * sizeof(float));
}

float getErr(float* toChange, float* givenGrad){
	float t = *toChange;
	(*toChange) += h;
	float new_loss = 0;
	net->fit_network(input, target, 1, &new_loss);
	*toChange = t;

	float calc = (new_loss - original_loss) / h;
	return std::abs(calc - *givenGrad);
}