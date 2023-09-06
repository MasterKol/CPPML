#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "../layer.hpp"
#include "../network.hpp"

namespace CPPML {

class Network;

/*
 * Special layer type used exclusively for input into a network.
 * Can be extended if needed.
 */
class Input : public Layer {
public:
	// creates a new network input and adds it
	// to the given network if one is provided
	Input(Shape input_shape, Network* net=nullptr);

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "Input";}

private:
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