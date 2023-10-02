#ifndef ACTIVATION_LAYER_HEADER
#define ACTIVATION_LAYER_HEADER

#include "../layer.hpp"
#include "../activation_func.hpp"

namespace CPPML {

class ActivationLayer : public Layer {
	const ActivationFunc* act;

	template<typename... Ts>
	ActivationLayer(const ActivationFunc* const activation, Ts... input_layers) :
				Layer(input_layers...), act(activation){}

	virtual void populate(float* params, float* gradients){}

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "Activation";}
private:
	// performs this layer's computation reading from the input
	// and writing to the output
	virtual void compute(float* input, float* output, float* intermediate_buffer);

	// takes in previous layer's change and calculates the change
	// of its inputs, WRITE TO inpt_change. inpt_change will always
	// be a copy so just write to it, adding is done externally
	// output, out_change, and intermediate can be changed because 
	// they will not be used downstream
	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);

	// sets up a layer given its inputs are already
	// compiled, only need to set i/o size and intermediate_num
	// returns true if layer is an input layer, false otherwise
	virtual bool compile_();
};


} // namespace CPPML

#endif