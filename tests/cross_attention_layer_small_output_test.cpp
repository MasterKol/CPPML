#include "layer_test.hpp"
#include "../include/Layers/cross_attention.hpp"

#include <iostream>

#include "../include/shape.hpp"
#include "../include/activation.hpp"

int main(){
	net = new CPPML::Network(CPPML::MAE);
	CPPML::Input* l1 = new CPPML::Input(CPPML::Shape(20, 30), net);
	CPPML::Input* l2 = new CPPML::Input(CPPML::Shape(20, 30), net);
	new CPPML::CrossAttention(1, 20, 20, 10, {l1}, {l2});
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}