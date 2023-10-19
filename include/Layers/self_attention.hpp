#ifndef SELF_ATTENTION_HEADER
#define SELF_ATTENTION_HEADER

#include "../layer.hpp"

namespace CPPML {

/*
 * Implements the self attention mechanism
 * Output dims match input dims by default,
 * output width can be changed.
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

	/// @param num_heads_ number of attention heads
	/// @param internal_size_ feature embed size, smaller is faster but less expressive
	/// @param output_width width of the output, height is the same as input
	/// @param input_width width to cast inputs to, original shapes are ignored
	/// @param input_layers vararg, inputs to this layer, all widths should match
	template<typename... Ts>
	SelfAttention(int num_heads_, int internal_size_, int output_width, int input_width, Ts... input_layers) : Layer(input_layers...){
		num_heads = num_heads_;
		internal_size = internal_size_;
		output_shape.w(output_width);
		input_shape.w(input_width);
	}

	/// @param num_heads_ number of attention heads
	/// @param internal_size_ feature embed size, smaller is faster but less expressive
	/// @param output_width width of the output, height is the same as input
	/// @param input_layers vararg, inputs to this layer, all widths should match
	template<typename... Ts>
	SelfAttention(int num_heads_, int internal_size_, int output_width, Ts... input_layers) : Layer(input_layers...){
		num_heads = num_heads_;
		internal_size = internal_size_;
		output_shape.w(output_width);
	}

	/// @param num_heads_ number of attention heads
	/// @param internal_size_ feature embed size, smaller is faster but less expressive
	/// @param input_layers vararg, inputs to this layer, all widths should match
	template<typename... Ts>
	SelfAttention(int num_heads_, int internal_size_, Ts... input_layers) : Layer(input_layers...){
		num_heads = num_heads_;
		internal_size = internal_size_;
		output_shape.w(-1);
	}

	virtual void populate(float* params, float* gradients);
	virtual std::string get_type_name(){return "SelfAttention";}
private:
	virtual void compute(float* input, float* output, float* intermediate_buffer);

	virtual bool compile_();

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