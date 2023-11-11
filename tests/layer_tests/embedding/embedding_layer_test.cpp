#include "../layer_test.hpp"
#include "Layers/embedding.hpp"

#include <iostream>

#include "shape.hpp"
#include "activation_func.hpp"

const int num_classes = 10;

int to_test = 0;
void set_input_(float* input){
	input[0] = to_test;
	to_test = (to_test + 1) % num_classes;
}

int main(){
	CPPML::Random::time_seed();
	setup(new CPPML::Embedding(num_classes, 16), CPPML::Shape(1));

	for(int i = 0; i < num_classes; i++){
		retest();
		checkParameterGradients();
	}

	return 0;
}