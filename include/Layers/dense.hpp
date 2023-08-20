#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../layer.hpp"

#include "../activation.hpp"

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
	const Activation* activation;
	Dense(int nodes, const Activation* const activation);

	template<typename... Ts>
	Dense(int nodes, const Activation* const activation_, Ts... input_layers) : Layer(input_layers...){
		init(nodes, activation_);
	}

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "Dense";}
private:
	// initialize layer
	void init(int nodes, const Activation* const activation);

	// performs this layer's computation reading from the input
	// and writing to the output
	virtual void compute(float* input, float* output, float* intermediate_buffer);

	// sets up a layer given its inputs are already
	// compiled, only need to set i/o size and intermediate_num
	// returns true if layer is an input layer
	virtual bool compile_();

	// takes in previous layer's change and calculates the change
	// of its inputs, WRITE TO inpt_change. inpt_change will always
	// be a copy so just write to it, adding is done externally
	// output, out_change, and intermediate can be changed because 
	// they will not be used downstream
	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);
};

}

#endif