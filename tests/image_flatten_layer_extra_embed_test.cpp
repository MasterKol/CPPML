#include "layer_test.hpp"
#include "../include/Layers/image_flatten.hpp"

#include <iostream>

#include "../include/shape.hpp"
#include "../include/activation_func.hpp"

int main(){
	net = new CPPML::Network(CPPML::MAE);
	CPPML::Input* l1 = new CPPML::Input(CPPML::Shape(30, 30), net);
	CPPML::Input* l2 = new CPPML::Input(CPPML::Shape(20), net);
	new CPPML::ImageFlatten(6, 6, 4, 4, l1, l2);
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}