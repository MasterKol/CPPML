#ifndef DROPOUT_LAYER_HEADER
#define DROPOUT_LAYER_HEADER

#include <mutex>

#include "../layer.hpp"

namespace CPPML {

/*
 * Adds dropout to the given layer
 * output shape matches input shape
 */
class Dropout : public Layer {
public:
	double dropout_ratio;
private:
	std::mutex rng_mutex;
public:

	/// @brief 
	/// @param dropout_ratio percentage of inputs that are dropped to 0
	/// @param input_layer input to this layer
	Dropout(double dropout_ratio, Layer* input_layer=nullptr);

	virtual void populate(float* params, float* gradients){}

	virtual std::string get_type_name(){return "Dropout";}
private:
	virtual void compute(float* input, float* output, float* intermediate_buffer, bool training);

	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);

	virtual bool compile_();
};


} // namespace CPPML

#endif