#include <iostream>
#include <memory>
#include <cstring>
#include <cmath>

#include "Layers/activation.hpp"
#include "Layers/input.hpp"
#include "activation_func.hpp"
#include "random.hpp"
#include "network.hpp"
#include "layer.hpp"
#include "cost_func.hpp"

const int SIZE = 100;

float h = 1 / 256.0f;
float epsilon = 1e-2;

float *input, *target, *input_change;
CPPML::Network* net;

float original_loss = 0;

float getDerv(float*);

int main(){
	CPPML::Random::time_seed();

	net = new CPPML::Network(CPPML::CROSS_ENTROPY);

	CPPML::Layer* l = new CPPML::Input(CPPML::Shape(SIZE), net);
	new CPPML::ActivationLayer(CPPML::SOFTMAX, l);
	
	// make network
	net->compile(nullptr);

	// initialize memory
	input = new float[SIZE];
	target = new float[SIZE];
	float* out = new float[SIZE];
	input_change = new float[net->last_io_size];

	// fill input with random values
	CPPML::Random::fillGaussian(input, SIZE, 0, 1);
	net->eval(input, out);

	// fill target with random values and softmax it so get x > 0 and sum(X) = 1
	CPPML::Random::fillGaussian(target, SIZE, 0, 1);
	CPPML::SOFTMAX->f(target, target, SIZE);

	// compute input_change
	net->fit_network(input, target, nullptr, nullptr, input_change, &original_loss);

	// find avg error in input derivative
	float avg = 0;
	for(int i = 0; i < SIZE; i++){
		float calc = out[i] - target[i];
		float err = 0;
		if(calc != 0){
			err = std::abs((calc - input_change[i]) / calc);
		}else if(input_change[i] != 0){
			err = input_change[i] * 100000;
		}
		avg += err;

		std::cerr << i << ", " << err << ", " << calc << ", " << input_change[i] << "\n";
	}
	avg /= SIZE;

	std::cerr << "Input derivative avg error = " << avg * 100 << "%\n";
	if(avg > epsilon || avg < 0 || isnan(avg)){
		exit(-1);
	}

	return 0;
}