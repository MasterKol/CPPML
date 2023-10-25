#include "../layer_test.hpp"
#include "Layers/image_flatten.hpp"

#include <iostream>
#include <cassert>
#include <iomanip>

#include "Layers/self_attention.hpp"
#include "shape.hpp"

int main(){
	net = new CPPML::Network(CPPML::MAE);
	CPPML::Input* in = new CPPML::Input(CPPML::Shape(12, 12), net);
	CPPML::ImageFlatten* flt = new CPPML::ImageFlatten(3, 3, 0, 0, in);
	CPPML::Layer* l = new CPPML::SelfAttention(1, 9, 9, flt);
	new CPPML::ImageDeFlatten(flt, l);
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}