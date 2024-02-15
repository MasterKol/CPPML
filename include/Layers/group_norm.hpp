#ifndef GROUP_NORM_LAYER_HEADER
#define GROUP_NORM_LAYER_HEADER

#include "../layer.hpp"

namespace CPPML {

/*
 * 
 */
class GroupNorm : public Layer {
public:
	const int num_groups;

	float* params; // [beta0, gamma0, beta1, gamma1, ...]
	float* gradients; // [beta0, gamma0, beta1, gamma1, ...]

	/// @param num_groups number of classes to learn
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	GroupNorm(int num_groups, Ts... input_layers) : Layer(input_layers...), num_groups(num_groups){}

	virtual void populate(float* params, float* gradients);

	virtual std::string get_type_name(){return "GroupNorm";}
private:
	//void norm_group(float* input, float* output, float* inter, int size);

	virtual void compute(float* input, float* output, float* intermediate_buffer, bool training);

	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);

	virtual bool compile_();
};


} // namespace CPPML

#endif