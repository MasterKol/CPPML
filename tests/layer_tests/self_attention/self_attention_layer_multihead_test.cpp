#include "../layer_test.hpp"
#include "Layers/self_attention.hpp"

#include <iostream>

#include "shape.hpp"

int main(){
	setup(new CPPML::SelfAttention(5, 10),
			CPPML::Shape(10, 20));

	checkInputGradients();
	checkParameterGradients();

	return 0;
}