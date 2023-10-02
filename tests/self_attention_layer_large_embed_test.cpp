#include "layer_test.hpp"
#include "../include/Layers/self_attention.hpp"

#include <iostream>

#include "../include/shape.hpp"

int main(){
	setup(new CPPML::SelfAttention(2, 30),
			CPPML::Shape(20, 30));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}