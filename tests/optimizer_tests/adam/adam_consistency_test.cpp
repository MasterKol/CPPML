#include <iostream>

#include "shape.hpp"
#include "network.hpp"
#include "random.hpp"
#include "optimizer.hpp"
#include "Optimizers/adam.hpp"

const int num_params = 1000;
int seed = 0;

CPPML::Network* net;
//CPPML::Adam* opt;

void get_outputs(float* prms, int l);

int main(){
	seed = CPPML::Random::time_seed();
	CPPML::Random::time_seed();

	net = new CPPML::Network(nullptr);

	net->num_params = num_params;
	net->gradients = new float[num_params];
	net->params = new float[num_params];

	const int num = 10;
	float* first_prms = new float[num_params * num];
	get_outputs(first_prms, num);

	float* new_prms = new float[num_params * num];

	for(int n = 0; n < 1000; n++){
		get_outputs(new_prms, num);
		for(int i = 0; i < num_params * num; i++){
			if(first_prms[i] != new_prms[i]){
				std::cerr << "Got: " << new_prms[i] << ", expected: " << first_prms[i] << std::endl;
				exit(-1);
			}
		}
	}
}

void get_outputs(float* prms, int l){
	CPPML::Adam* opt = new CPPML::Adam();
	opt->compile(net);

	CPPML::Random::rand_seed(seed);
	CPPML::Random::fillGaussian(net->params, num_params, 0, 1);

	float* out = prms;

	for(int i = 0; i < l; i++){
		CPPML::Random::fillGaussian(net->params, num_params, 0, 1);

		opt->update_params();

		memcpy(out, net->params, num_params * sizeof(float));

		out += num_params;
	}

	delete opt;
}