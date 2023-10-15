#include "../layer_test.hpp"
#include "Layers/maxpooling2d.hpp"

#include <iostream>

#include "shape.hpp"

int main(){
	setup(new CPPML::MaxPooling2d(2, 2),
			CPPML::Shape(20, 20, 3));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}