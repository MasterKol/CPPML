#include "../layer_test.hpp"

#include "shape.hpp"
#include "Layers/conv2d.hpp"

int main(){	
	net = new CPPML::Network(CPPML::MSE);
	CPPML::Input* l1 = new CPPML::Input(CPPML::Shape(10, 10, 2), net);
	CPPML::Input* l2 = new CPPML::Input(CPPML::Shape(100), net);
	new CPPML::Conv2d(3, 3, 3, CPPML::LINEAR, 1, l1, l2);
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}