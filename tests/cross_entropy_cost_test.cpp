#include <iostream>
#include <memory>
#include <cstring>
#include <cmath>

#include "../include/cost_func.hpp"
#include "../include/random.hpp"

const int SIZE = 100;
const float epsilon = 1e-4;

const int DERV_SIZE = 100;
const float derv_h = 1e-6;
const float derv_epsilon = 1e-2;

void normalize(float* v, int len){
	float total = 0;
	for(int i = 0; i < len; i++)
		total += v[i];
	for(int i = 0; i < len; i++)
		v[i] /= total;
}

int main(){
	std::unique_ptr<float[]> x(new float[SIZE]);
	std::unique_ptr<float[]> x_cpy(new float[SIZE]);
	std::unique_ptr<float[]> y(new float[SIZE]);
	std::unique_ptr<float[]> y_cpy(new float[SIZE]);
	std::unique_ptr<float[]> out(new float[SIZE]);

	CPPML::Random::time_seed();

	CPPML::Random::fillRand(x.get(), SIZE, 0, 1);
	normalize(x.get(), SIZE);
	CPPML::Random::fillRand(y.get(), SIZE, 0, 1);
	normalize(y.get(), SIZE);

	memcpy(x_cpy.get(), x.get(), SIZE * sizeof(float));
	memcpy(y_cpy.get(), y.get(), SIZE * sizeof(float));

	/*** test function ***/
	float cost = CPPML::CROSS_ENTROPY->get_cost(x.get(), y.get(), SIZE);
	
	// make sure that input didn't change
	for(int i = 0; i < SIZE; i++){
		if(x[i] != x_cpy[i]){
			std::cerr << "CROSS_ENTROPY->get_cost changed input array x which is not allowed!\n";
			exit(-1);
		}

		if(y[i] != y_cpy[i]){
			std::cerr << "CROSS_ENTROPY->get_cost changed input array y which is not allowed!\n";
			exit(-1);
		}
	}

	// make sure function outputted correct value
	float total = 0;
	for(int i = 0; i < SIZE; i++){
		if(x[i] <= 0 || y[i] <= 0)
			continue;
		total -= y[i] * log(x[i] + 1e-10);
	}

	if(abs(cost - total) > epsilon){
		std::cerr << "An error occured in function calculation:\n";
		std::cerr << "Expected out: " << total
			<< ", Real out: " << cost << "\n";
		exit(-1);
	}

	/*** test function derivative ***/
	CPPML::CROSS_ENTROPY->get_cost_derv(x.get(), y.get(), out.get(), SIZE);
	
	// make sure that input didn't change
	for(int i = 0; i < SIZE; i++){
		if(x[i] != x_cpy[i]){
			std::cerr << "CROSS_ENTROPY->get_cost_derv changed input array x which is not allowed!\n";
			exit(-1);
		}

		if(y[i] != y_cpy[i]){
			std::cerr << "CROSS_ENTROPY->get_cost_derv changed input array y which is not allowed!\n";
			exit(-1);
		}
	}

	/*** Test Derivative Numerically ***/
	for(int i = 0; i < DERV_SIZE; i++){
		float calc = -y[i] / (x[i] + 1e-10);

		std::cerr << abs((calc - out[i]) / out[i]) << "\n";

		if(abs((calc - out[i]) / out[i]) > derv_epsilon){
			std::cerr << "Calculated derivative does not match actual response:\n";
			std::cerr << "x: " << x[i] << ", y: " << y[i]
				<< ", Expected out: " << calc
				<< ", Real out: " << out[i] << ", diff: " << abs(out[i] - calc) << "\n";
			exit(-1);
		}
	}

	return 0;
}