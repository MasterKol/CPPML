#include "../layer_test.hpp"
#include "Layers/dense.hpp"

#include <iostream>

#include "shape.hpp"
#include "activation_func.hpp"

int main(){
	setup(new CPPML::Dense(25, CPPML::LINEAR, false), CPPML::Shape(30));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}