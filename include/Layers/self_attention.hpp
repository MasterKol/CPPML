#ifndef SELF_ATTENTION_HEADER
#define SELF_ATTENTION_HEADER

#include "../layer.hpp"

namespace CPPML {

/*
 * 
 * DON'T USE add_input FOR THIS LAYER, USE add_QV / add_K
 */
class SelfAttention : public Layer {
public:
	int num_heads;
	int internal_size;

	// (internal_size, input_shape.w, num_heads)
	float *q_mat, *v_mat, *k_mat;
	// (output_shape.w, num_heads * internal_size)
	float *z_mat;

	// (internal_size, input_shape.w, num_heads)
	float *q_grads, *v_grads, *k_grads;
	// (output_shape.w, num_heads * internal_size)
	float *z_grads;

	template<typename... Ts>
	SelfAttention(int num_heads_, int internal_size_, int output_width, int input_width, Ts... input_layers) : Layer(input_layers...){
		num_heads = num_heads_;
		internal_size = internal_size_;
		output_shape.w = output_width;
		input_shape.w = input_width;
	}

	template<typename... Ts>
	SelfAttention(int num_heads_, int internal_size_, int output_width, Ts... input_layers) : Layer(input_layers...){
		num_heads = num_heads_;
		internal_size = internal_size_;
		output_shape.w = output_width;
	}

	template<typename... Ts>
	SelfAttention(int num_heads_, int internal_size_, Ts... input_layers) : Layer(input_layers...){
		num_heads = num_heads_;
		internal_size = internal_size_;
		output_shape.w = -1;
	}

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "SelfAttention";}
	
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

	// evaluates one attention head given a pointer to the weight matrices
	// associated with the head and pointers to auxiliary memory.
	// Can be made to not calculate O if desired
	inline void attention_head(float* input,
		float* qm, float* km, float* vm, float* zm,
		float* Q, float* K, float* V, float* QKT,
		float* Z, float* O, const float norm_factor, bool calculateO);

	// Calculates the derivative of Q, V, and K
	// buff must be of size Iw * max(in_sz, Ih)
	inline void QVK_Derv(float* Pw, float* dPw, float* dP, float* inputT, float* dIn, float* buff, bool dP_transposed=false);
};

}

#endif