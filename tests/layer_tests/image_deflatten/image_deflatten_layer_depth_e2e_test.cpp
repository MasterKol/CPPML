#include "../layer_test.hpp"
#include "Layers/image_flatten.hpp"

#include <iostream>
#include <cassert>
#include <iomanip>

#include "Layers/self_attention.hpp"
#include "Layers/conv2d.hpp"
#include "shape.hpp"

int main(){
	net = new CPPML::Network(CPPML::MAE);
	CPPML::Input* in = new CPPML::Input(CPPML::Shape(12, 12, 2), net);
	CPPML::ImageFlatten* flt = new CPPML::ImageFlatten(3, 3, 2, 2, in);
	CPPML::Layer* l = new CPPML::SelfAttention(1, 9, 18, flt);
	// CPPML::Layer* l = new CPPML::Conv2d(3, 3, 1, 1, CPPML::ELU, flt);
	new CPPML::ImageDeFlatten(flt, l);
	setup();

	checkInputGradients();
	checkParameterGradients();

	return 0;
}