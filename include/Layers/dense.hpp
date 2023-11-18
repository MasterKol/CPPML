#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../layer.hpp"
#include "../activation_func.hpp"

namespace CPPML {

/*
 * Takes in 1d array of data and returns 1d array of given length. 
 * Runs data through given activation function before moving on.
 */
class Dense : public Layer {
public:
	float *weights, *biases;
	float *weight_grads, *bias_grads;
	int num_weights, num_biases;
	const ActivationFunc* activation;
	const bool use_bias;

	/// @param nodes  number of nodes, size of output
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Dense(int nodes, Ts... input_layers) : Layer(input_layers...), activation(nullptr), use_bias(true){
		output_shape = Shape(nodes);
	}

	/// @param nodes  number of nodes, size of output
	/// @param activation activation function to run after processing
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Dense(int nodes, const ActivationFunc* const activation, Ts... input_layers) : Layer(input_layers...), activation(activation), use_bias(true){
		output_shape = Shape(nodes);
	}

	/// @param nodes  number of nodes, size of output
	/// @param activation activation function to run after processing
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Dense(int nodes, const ActivationFunc* const activation, bool use_bias, Ts... input_layers) : Layer(input_layers...), activation(activation), use_bias(use_bias){
		output_shape = Shape(nodes);
	}

	virtual void populate(float* params, float* gradients);

	virtual std::string get_type_name(){return "Dense";}
private:
	virtual void compute(float* input, float* output, float* intermediate_buffer, bool training);

	virtual bool compile_();

	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);
};

}

#endif