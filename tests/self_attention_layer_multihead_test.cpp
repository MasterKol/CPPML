#include "layer_test.hpp"
#include "../include/Layers/self_attention.hpp"

#include <iostream>

#include "../include/shape.hpp"

int main(){
	setup(new CPPML::SelfAttention(5, 20),
			CPPML::Shape(20, 30));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}