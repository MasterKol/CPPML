#include "layer_test.hpp"
#include "../include/Layers/image_flatten.hpp"

#include <iostream>

#include "../include/shape.hpp"
#include "../include/activation.hpp"

int main(){
	net = new CPPML::Network(CPPML::MAE);
	CPPML::Input* l = new CPPML::Input(CPPML::Shape(30, 30), net);
	new CPPML::ImageFlatten(6, 6, 4, 4, l);
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}