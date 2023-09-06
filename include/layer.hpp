#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <mutex>
#include <cassert>
#include <string>

#include "shape.hpp"

namespace CPPML {

/*
 * Layer interface, all layers in a network extend this.
 * Layer* can be used for all types of layers.
 */
class Layer{
public:
	Shape input_shape, output_shape;
	std::vector<Layer*> inputs, outputs;

	// number of variable parameters this layer has
	int num_params;

	// size of intermediate values that need to be
	// stored during training
	int intermediate_num;

	// index in intermediate buffer
	int intermediate_index;
	// index of start of outputs in output buffer
	int output_index;

	// mutex to protect gradients while they are being modified
	std::mutex gradient_mutex;

	std::string name;

	template<typename... Ts>
	Layer(Ts... input_layers){
		num_params = 0;
		output_index = 0;
		input_shape = Shape(-1);
		intermediate_num = 0;
		intermediate_index = 0;
		name = "";

		(add_input(input_layers), ...);
	}

	// set name of layer
	Layer* set_name(std::string name_){ name = name_; return this; }

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name() = 0;

	// adds given layer as an output to this layer
	// e.g. secondLayer.add_input(firstLayer)
	void add_input(Layer* layer);

	// returns the layer that represents the output
	// of this layer, usually 'this' but in some cases
	// may be different (defaults to 'this')
	Layer* get_output();

	// call for use by network, sets up input and output
	// for this layer to read and write from. If there are
	// multiple inputs it collects them together into
	// temporary memory
	void process(float* io_buffer, float* intermediate_buffer=nullptr);

	// calls compile_ for each layer which sets up the layer
	// given its inputs, this redirect just does boilerplate
	// calculations
	void compile(int buffer_index, int inter_index);

	// sets up inputs for get_change and add_gradients, collects
	// input and output changes, inputs and outputs, and sets 
	// offset for intermediate buffer finally writes output changes 
	// to their proper place in the change buffer
	void backpropagate(float* change_buffer, float* io_buffer,
					   float* intermediate_buffer);

	// gets pointer to parameter memory from network and
	// fills it with initial params. Also stores pointer to
	// gradients for the layer's parameters
	virtual void populate(float* params, float* gradients) = 0;
private:
	// initializer function
	void init();

	// collects inputs from the io buffer and writes them into
	// the provided array input should be at least
	// input_shape.size * sizeof(float) bytes long
	void collect_inputs(float* io_buffer, float* input);

	// performs this layer's computation reading from the input
	// and writing to the output
	virtual void compute(float* input, float* output, 
						 float* intermediate_buffer) = 0;

	// sets up a layer given its inputs are already
	// compiled, only need to set i/o size and intermediate_num
	// returns true if layer is an input layer, false otherwise
	virtual bool compile_() = 0;

	// takes in previous layer's change and calculates the change
	// of its inputs, WRITE TO inpt_change. inpt_change will always
	// be a copy so just write to it, adding is done externally
	// output, out_change, and intermediate can be changed because 
	// they will not be used downstream. Also calculates gradients
	// for this layer's params and adds them to its gradient buffer
	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate) = 0;
};

}

#endif