#include "../layer_test.hpp"
#include "Layers/dense.hpp"

#include <iostream>

#include "shape.hpp"
#include "activation_func.hpp"

int main(){
	setup(new CPPML::Dense(25, CPPML::LINEAR), CPPML::Shape(30));

	//net->print_summary();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}