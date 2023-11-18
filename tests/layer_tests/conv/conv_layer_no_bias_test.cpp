#include "../layer_test.hpp"
#include "Layers/conv2d.hpp"

#include <iostream>

#include "shape.hpp"
#include "activation_func.hpp"

int main(){
	setup(new CPPML::Conv2d(3, 3, 5, CPPML::ELU, 1, false),
				CPPML::Shape(20, 20, 3));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}