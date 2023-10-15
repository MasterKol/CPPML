#include <iostream>
#include <memory>
#include <cstring>
#include <cmath>

#include "cost_func.hpp"
#include "random.hpp"

const int SIZE = 1000;
const float epsilon = 1e-4;

const int DERV_SIZE = 1000;
const float derv_h = 1e-3;
const float derv_epsilon = 1e-3;

int main(){
	std::unique_ptr<float[]> x(new float[SIZE]);
	std::unique_ptr<float[]> x_cpy(new float[SIZE]);
	std::unique_ptr<float[]> y(new float[SIZE]);
	std::unique_ptr<float[]> y_cpy(new float[SIZE]);
	std::unique_ptr<float[]> out(new float[SIZE]);

	CPPML::Random::time_seed();

	CPPML::Random::fillGaussian(x.get(), SIZE, 0, 1);
	CPPML::Random::fillGaussian(y.get(), SIZE, 0, 1);

	memcpy(x_cpy.get(), x.get(), SIZE * sizeof(float));
	memcpy(y_cpy.get(), y.get(), SIZE * sizeof(float));

	/*** test function ***/
	float cost = CPPML::HUBER->get_cost(x.get(), y.get(), SIZE);
	
	// make sure that input didn't change
	for(int i = 0; i < SIZE; i++){
		if(x[i] != x_cpy[i]){
			std::cerr << "HUBER->get_cost changed input array x which is not allowed!\n";
			exit(-1);
		}

		if(y[i] != y_cpy[i]){
			std::cerr << "HUBER->get_cost changed input array y which is not allowed!\n";
			exit(-1);
		}
	}

	// make sure function outputted correct value
	float total = 0;
	for(int i = 0; i < SIZE; i++){
		float t = std::abs(x[i] - y[i]);
		total += (t < 1.0f) ? t * t * 0.5 : t - 0.5;
	}
	total /= SIZE;

	if(abs(cost - total) > epsilon){
		std::cerr << "An error occured in function calculation:\n";
		std::cerr << "Expected out: " << total
			<< ", Real out: " << cost << "\n";
		exit(-1);
	}

	/*** test function derivative ***/
	CPPML::HUBER->get_cost_derv(x.get(), y.get(), out.get(), SIZE);
	
	// make sure that input didn't change
	for(int i = 0; i < SIZE; i++){
		if(x[i] != x_cpy[i]){
			std::cerr << "HUBER->get_cost_derv changed input array x which is not allowed!\n";
			exit(-1);
		}

		if(y[i] != y_cpy[i]){
			std::cerr << "HUBER->get_cost_derv changed input array y which is not allowed!\n";
			exit(-1);
		}
	}

	/*** Test Derivative Numerically ***/
	CPPML::HUBER->get_cost_derv(x.get(), y.get(), out.get(), DERV_SIZE);
	float baseCost = CPPML::HUBER->get_cost(x.get(), y.get(), DERV_SIZE);

	for(int i = 0; i < DERV_SIZE; i++){
		x[i] += derv_h;
		float new_cost = CPPML::HUBER->get_cost(x.get(), y.get(), DERV_SIZE);
		x[i] -= derv_h;
		float calc = (new_cost - baseCost) / derv_h;

		if(abs(out[i] - calc) > derv_epsilon){
			std::cerr << "Calculated derivative does not match actual response:\n";
			std::cerr << "x: " << x[i] << ", y: " << y[i]
				<< ", Expected out: " << calc
				<< ", Real out: " << out[i] << ", diff: " << abs(out[i] - calc) << "\n";
			exit(-1);
		}
	}

	return 0;
}