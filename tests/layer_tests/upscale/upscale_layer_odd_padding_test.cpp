#include "../layer_test.hpp"
#include "Layers/upscale2d.hpp"

#include <iostream>

#include "shape.hpp"

int main(){
	setup(new CPPML::Upscale2d(2, 1),
			CPPML::Shape(20, 20, 3));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}