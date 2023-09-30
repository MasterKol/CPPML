#include "layer_test.hpp"
#include "../include/Layers/cross_attention.hpp"

#include <iostream>

#include "../include/shape.hpp"
#include "../include/activation.hpp"

int main(){
	net = new CPPML::Network(CPPML::MAE);
	// Q inputs
	CPPML::Input* q1 = new CPPML::Input(CPPML::Shape(20, 30), net);
	CPPML::Input* q2 = new CPPML::Input(CPPML::Shape(20, 30), net);

	// VK inputs
	CPPML::Input* vk1 = new CPPML::Input(CPPML::Shape(20, 30), net);
	CPPML::Input* vk2 = new CPPML::Input(CPPML::Shape(20, 30), net);
	new CPPML::CrossAttention(1, 20, 20, 20, {q1, q2}, {vk1, vk2});
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}