#include "layer_test.hpp"
#include "../include/Layers/conv2d.hpp"

#include <iostream>

#include "../include/shape.hpp"
#include "../include/activation.hpp"

CPPML::Shape input_shape = CPPML::Shape(20, 20, 3);

int main(){
	setup(new CPPML::Conv2d(3, 3, 5, CPPML::LINEAR, 0));

	/**** CHECK THAT INPUT GRADIENTS MATCH ****/

	for(int i = 0; i < input_shape.size(); i++){
		float calc = getErr(input + i, input_change + i);
		if(calc > epsilon){
			std::cerr << "Incorrect derivative of inputs, got: " 
				<< input_change[i] << ", but expected: " << calc << "\n";
			return 1;
		}
	}

	/**** CHECK THAT PARAMETER GRADIENTS MATCH ****/

	for(int i = 0; i < net->num_params; i++){
		float calc = getErr(net->params + i, gradients + i);
		if(calc > epsilon){
			std::cerr << "Incorrect derivative of parameters, got: " 
				<< gradients[i] << ", but expected: " << calc << "\n";
			return 1;
		}
	}

	return 0;
}