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
	/// @brief Creates new Input and automatically adds it as input to a network if one is provided.
	/// @param input_shape shape of input into this layer, equal to output shape
	/// @param net *optional* adds this layer as an input to the given layer
	Input(Shape input_shape, Network* net=nullptr);
	
	virtual void populate(float* params, float* gradients);
	virtual std::string get_type_name(){return "Input";}

private:
	virtual void compute(float* input, float* output, float* intermediate_buffer);
	virtual bool compile_();
	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);
};

}

#endif