#include <iostream>
#include <memory>
#include <cstring>
#include <cmath>

#include "activation_func.hpp"
#include "random.hpp"

const int SIZE = 1000;
const float epsilon = 1e-6;

int main(){
	std::unique_ptr<float[]> in(new float[SIZE]);
	std::unique_ptr<float[]> in_cpy(new float[SIZE]);
	std::unique_ptr<float[]> out(new float[SIZE]);
	std::unique_ptr<float[]> grad(new float[SIZE]);
	std::unique_ptr<float[]> out_buff(new float[SIZE]);

	CPPML::Random::time_seed();

	CPPML::Random::fillGaussian(in.get(), SIZE, 0, 1);

	memcpy(in_cpy.get(), in.get(), SIZE * sizeof(float));

	for(int i = 0; i < SIZE; i++) grad[i] = 1;

	/*** test function ***/
	CPPML::SIGMOID->f(in.get(), out.get(), SIZE);
	
	// make sure that input didn't change
	for(int i = 0; i < SIZE; i++){
		if(in[i] != in_cpy[i]){
			std::cerr << "SIGMOID->f changed input array which is not allowed!\n";
			exit(-1);
		}
	}

	// make sure function outputted correct value
	for(int i = 0; i < SIZE; i++){
		float calc = 1.0f / (1.0f + std::exp(-in[i]));
		if(abs(out[i] - calc) > epsilon){
			std::cerr << "An error occured in function calculation:\n";
			std::cerr << "Input: " << in[i] << ", Expected out: " << calc
				<< ", Real out: " << out[i] << "\n";
			exit(-1);
		}
	}

	/*** test function derivative ***/
	CPPML::SIGMOID->df(in.get(), out.get(), out_buff.get(), grad.get(), SIZE);
	
	// make sure that input didn't change
	for(int i = 0; i < SIZE; i++){
		if(in[i] != in_cpy[i]){
			std::cerr << "SIGMOID->df changed input array which is not allowed!\n";
			exit(-1);
		}
	}

	// make sure function outputted correct value
	for(int i = 0; i < SIZE; i++){
		float calc = 1.0f / (1.0f + exp(-in[i]));
		calc = exp(-in[i]) * calc * calc;
		if(abs(out[i] - calc) > epsilon){
			std::cerr << "An error occured in derivative function calculation:\n";
			std::cerr << "Input: " << in[i] << ", Expected out: " << calc
				<< ", Real out: " << out[i] << "\n";
			exit(-1);
		}
	}

	return 0;
}