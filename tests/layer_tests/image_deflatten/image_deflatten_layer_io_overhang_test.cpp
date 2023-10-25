#include "../layer_test.hpp"
#include "Layers/image_flatten.hpp"

#include <iostream>

#include "shape.hpp"

int main(){
	net = new CPPML::Network(CPPML::MSE);
	CPPML::Input* in = new CPPML::Input(CPPML::Shape(26, 26), net);
	CPPML::ImageFlatten* l = new CPPML::ImageFlatten(3, 3, 0, 0, in);
	new CPPML::ImageDeFlatten(l, l);
	setup();

	assert(input_length == output_length);

	for(int i = 0; i < output_length; i++){
		if(input[i] != output[i]){
			std::cerr << "input and output don't match\n";
			std::cerr << i << ", " << input[i] << ", " << output[i] << std::endl;
			exit(-1);
		}
	}

	checkInputGradients();

	return 0;
}