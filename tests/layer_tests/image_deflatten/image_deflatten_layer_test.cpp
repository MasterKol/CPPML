#include "../layer_test.hpp"
#include "Layers/image_flatten.hpp"

#include <iostream>

#include "shape.hpp"

int main(){
	net = new CPPML::Network(CPPML::MAE);
	CPPML::Input* l = new CPPML::Input(CPPML::Shape(9, 9), net);
	new CPPML::ImageDeFlatten(3, 3, 9, 9, 1, l);
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}