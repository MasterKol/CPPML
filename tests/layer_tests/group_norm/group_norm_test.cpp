#include "../layer_test.hpp"
#include "Layers/group_norm.hpp"

#include <iostream>
#include <iomanip>

#include "shape.hpp"

int main(){
	setup(new CPPML::GroupNorm(1), CPPML::Shape(3, 3, 2)); //1456400379, 1304117872, -1856369632

	// std::cout << std::setprecision(15) << std::flush;

	// for(int i = 0; i < 2; i++){
	// 	std::cout << net->params[i] << ", ";
	// }
	// std::cout << std::endl;

	// std::cout << std::endl;
	// for(int i = 0; i < 18;){
	// 	std::cout << input[i++] << ", ";
	// }
	
	// std::cout << std::endl << std::endl;
	// for(int i = 0; i < 18;){
	// 	std::cout << output[i++] << ", ";
	// }

	// std::cout << std::endl << std::endl;
	// for(int i = 0; i < 0;){
	// 	std::cout << input_change[i++] << ", ";
	// }

	// std::cout << std::endl << std::endl;
	// for(int i = 18; i < 36;){
	// 	std::cout << input_change[i++] << ", ";
	// }

	// std::cout << std::endl << std::endl;

	// input[34] -= h;
	// net->eval(input, output);

	// float* input_copy = new float[input_length];
	// float* output_copy = new float[output_length];
	// float* param_copy = new float[net->num_params];

	// memcpy(input_copy, input, input_length * sizeof(float));
	// memcpy(output_copy, output, output_length * sizeof(float));
	// memcpy(param_copy, net->params, net->num_params * sizeof(float));

	// for(int i = 0; i < 20; i++){
	// 	net->eval(input, output);
	// 	for(int j = 0; j < net->num_params; j++){
	// 		if(param_copy[j] != net->params[j]){
	// 			std::cout << "params not equal: " << j << ", " << param_copy[j] << ", " << net->params[j] << std::endl;
	// 			exit(-1);
	// 		}
	// 	}
	// 	for(int j = 0; j < input_length; j++){
	// 		if(input_copy[j] != input[j]){
	// 			std::cout << "inputs not equal: " << j << ", " << input_copy[j] << ", " << input[j] << std::endl;
	// 			exit(-1);
	// 		}

	// 		if(output_copy[j] != output[j]){
	// 			std::cout << "outputs not equal: " << j << ", " << output_copy[j] << ", " << output[j] << std::endl;
	// 			exit(-1);
	// 		}
	// 	}
	// 	std::cout << net->cost_func->get_cost(output, target, output_length) << std::endl;
	// }


	// net->print_summary();

	checkInputGradients();
	checkParameterGradients();

	/*const int N = 4;
	float g[] = {0, 0, 0, 0};
	float g2[N];

	float mean, stdev;

	memcpy(g2, g, N * sizeof(float));
	vDSP_normalize(g2, 1, g2, 1, &mean, &stdev, N);

	std::cout << mean << ", " << stdev << std::endl;

	for(int i = 0; i < N; i++){
		std::cout << g2[i] << ", ";
	}
	std::cout << std::endl;

	memcpy(g2, g, N * sizeof(float));
	CPPML::vDSP_normalize(g2, 1, g2, 1, &mean, &stdev, N);

	std::cout << mean << ", " << stdev << std::endl;

	for(int i = 0; i < N; i++){
		std::cout << g2[i] << ", ";
	}
	std::cout << std::endl;*/

	return 0;
}