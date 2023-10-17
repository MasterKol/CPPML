#include "../layer_test.hpp"
#include "Layers/cross_attention.hpp"

#include <iostream>

#include "shape.hpp"

int main(){
	epsilon = 2e-2;
	
	net = new CPPML::Network(CPPML::MAE);
	CPPML::Input* l1 = new CPPML::Input(CPPML::Shape(10, 20), net);
	CPPML::Input* l2 = new CPPML::Input(CPPML::Shape(10, 20), net);
	new CPPML::CrossAttention(1, 10, 10, 10, {l1}, {l2});
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}