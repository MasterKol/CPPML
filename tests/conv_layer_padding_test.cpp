#include "layer_test.hpp"
#include "../include/Layers/conv2d.hpp"

#include <iostream>

#include "../include/shape.hpp"
#include "../include/activation.hpp"

int main(){
	setup(new CPPML::Conv2d(3, 3, 5, CPPML::LINEAR, 1), 
				CPPML::Shape(20, 20, 3));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}