#include "../layer_test.hpp"
#include "Layers/dense.hpp"

#include <iostream>

#include "shape.hpp"
#include "activation_func.hpp"

int main(){
	setup(new CPPML::Dense(25, CPPML::LINEAR), CPPML::Shape(30));

	net->eval(input, output);

	float* output2 = new float[output_length];

	for(int i = 0; i < 1000; i++){
		net->eval(input, output2);

		for(int j = 0; j < output_length; j++){
			if(output2[j] != output[j]){
				std::cerr << "outputs not equal, " << output2[j] << ", " << output[j] << std::endl;
				exit(-1);
			}
		}
	}

	float* input_change2 = new float[net->last_io_size];

	for(int i = 0; i < 1000; i++){
		memset(net->gradients, 0, net->num_params * sizeof(float));
		memset(input_change2, 0, net->last_io_size * sizeof(float));

		float new_loss = 0;
		net->fit_network(input, target, nullptr, nullptr, input_change2, &new_loss);
		for(int j = 0; j < net->num_params; j++){
			if(gradients[j] != net->gradients[j]){
				std::cerr << "Param Gradients not equal, " << net->gradients[j] << ", " << gradients[j] << std::endl;
				exit(-1);
			}
		}

		for(int j = 0; j < net->last_io_size; j++){
			if(input_change[j] != input_change2[j]){
				std::cerr << "Gradients not equal, " << input_change2[j] << ", " << input_change[j] << std::endl;
				exit(-1);
			}
		}

		if(original_loss != new_loss){
			std::cerr << "Loss not equal, " << new_loss << ", " << original_loss << std::endl;
			exit(-1);
		}
	}

	return 0;
}