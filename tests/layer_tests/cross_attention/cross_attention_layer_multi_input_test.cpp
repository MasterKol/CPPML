#include "../layer_test.hpp"
#include "Layers/cross_attention.hpp"

#include <iostream>

#include "shape.hpp"

int main(){	
	net = new CPPML::Network(CPPML::MSE);
	// Q inputs
	CPPML::Input* q1 = new CPPML::Input(CPPML::Shape(10, 20), net);
	CPPML::Input* q2 = new CPPML::Input(CPPML::Shape(10, 20), net);

	// VK inputs
	CPPML::Input* vk1 = new CPPML::Input(CPPML::Shape(10, 20), net);
	CPPML::Input* vk2 = new CPPML::Input(CPPML::Shape(10, 20), net);
	new CPPML::CrossAttention(1, 10, 10, 10, {q1, q2}, {vk1, vk2});
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}