#include "../layer_test.hpp"
#include "Layers/image_flatten.hpp"

#include <iostream>
#include <cassert>
#include <iomanip>

#include "Layers/self_attention.hpp"
#include "shape.hpp"

int main(){
	net = new CPPML::Network(CPPML::MSE);
	CPPML::Input* in = new CPPML::Input(CPPML::Shape(11, 11, 2), net);
	CPPML::ImageFlatten* flt = new CPPML::ImageFlatten(3, 3, 0, 0, in);
	new CPPML::ImageDeFlatten(flt, flt);
	setup();

	assert(input_length == output_length);

	for(int i = 0; i < input_length; i++){
		if(input[i] != output[i]){
			std::cerr << "input and output don't match\n";
			std::cerr << i << ", " << input[i] << ", " << output[i] << std::endl;
			exit(-1);
		}
	}

	checkInputGradients();

	return 0;
}