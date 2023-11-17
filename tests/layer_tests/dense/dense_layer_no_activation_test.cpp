#include "../layer_test.hpp"

#include "Layers/dense.hpp"
#include "shape.hpp"

int main(){
	CPPML::Layer* l = new CPPML::Dense(25);
	new CPPML::Dense(50);
	setup(l, CPPML::Shape(30), 1);

	checkInputGradients();
	checkParameterGradients();

	return 0;
}