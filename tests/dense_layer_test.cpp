#include "layer_test.hpp"
#include "../include/Layers/dense.hpp"

#include <iostream>

#include "../include/shape.hpp"
#include "../include/activation.hpp"

int main(){
	setup(new CPPML::Dense(25, CPPML::LINEAR), CPPML::Shape(30));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}