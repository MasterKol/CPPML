#ifndef ACTIVATION_LAYER_HEADER
#define ACTIVATION_LAYER_HEADER

#include "../layer.hpp"
#include "../activation_func.hpp"

namespace CPPML {

/*
 * Applies the given activation function to input
 */
class ActivationLayer : public Layer {
public:
	const ActivationFunc* act;

	/// @brief Creates activation layer
	/// @param activation activation function to use
	/// @param input_layers vararg inputs to this layer
	template<typename... Ts>
	ActivationLayer(const ActivationFunc* const activation, Ts... input_layers) :
				Layer(input_layers...), act(activation){}

	virtual void populate(float* params, float* gradients){}

	virtual std::string get_type_name(){return "Activation";}
private:
	virtual void compute(float* input, float* output, float* intermediate_buffer);

	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);

	virtual bool compile_();
};


} // namespace CPPML

#endif