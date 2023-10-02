#include "layer_test.hpp"
#include "../include/Layers/upscale2d.hpp"

#include <iostream>

#include "../include/shape.hpp"

int main(){
	setup(new CPPML::Upscale2d(2, 2),
			CPPML::Shape(20, 20, 3));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}