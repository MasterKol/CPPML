#ifndef CROSS_ATTENTION_HEADER
#define CROSS_ATTENTION_HEADER

#include <vector>
#include <initializer_list>

#include "../layer.hpp"

namespace CPPML {

/*
 * DON'T USE add_input FOR THIS LAYER, USE add_VK / add_Q
 * Performs multi-head cross attention on two different sets
 * of inputs. Outputs with specified width and height of Q inputs.
 * Q and VK widths must match
 */
class CrossAttention : public Layer {
public:
	// number of attention heads
	int num_heads;
	// dimension that qk and v are projected into
	int qk_embed_size, v_embed_size;

	// (qk_embed_size, Q_shape.w, num_heads)
	float *q_mat, *q_grads;
	// (v_embed_size, VK_shape.w, num_heads)
	float* v_mat, *v_grads;
	// (qk_embed_size, VK_shape.w, num_heads)
	float *k_mat, *k_grads;
	// (output_width, v_embed_size * num_heads)
	float *z_mat, *z_grads;

	// shape of Q and VK *inputs*
	Shape Q_shape, VK_shape;

	// list of VK and Q layers
	std::vector<Layer*> VK_layers;
	std::vector<Layer*> Q_layers;

	/// @param num_heads number of attention heads
	/// @param qk_embed_size feature embed size of Q and K, smaller is faster but less expressive
	/// @param v_embed_size feature embed size of V, smaller is faster but less expressive
	/// @param Qs Q inputs, in a list captured by {}
	/// @param VKs VK inputs, in a list captured by {}
	CrossAttention(int num_heads, int qk_embed_size, int v_embed_size, std::initializer_list<Layer*> Qs={}, std::initializer_list<Layer*> VKs={}){
		init(num_heads, qk_embed_size, v_embed_size, -1, -1, Qs, VKs);
	}

	/// @param num_heads number of attention heads
	/// @param qk_embed_size feature embed size of Q and K, smaller is faster but less expressive
	/// @param v_embed_size feature embed size of V, smaller is faster but less expressive
	/// @param output_width output width of the layer
	/// @param Qs Q inputs, in a list captured by {}
	/// @param VKs VK inputs, in a list captured by {}
	CrossAttention(int num_heads, int qk_embed_size, int v_embed_size, int output_width, std::initializer_list<Layer*> Qs={}, std::initializer_list<Layer*> VKs={}){
		init(num_heads, qk_embed_size, v_embed_size, output_width, -1, Qs, VKs);
	}

	/// @param num_heads number of attention heads
	/// @param qk_embed_size feature embed size of Q and K, smaller is faster but less expressive
	/// @param v_embed_size feature embed size of V, smaller is faster but less expressive
	/// @param output_width output width of the layer
	/// @param Qwidth width to cast Q inputs to
	/// @param VKwidth width to cast VK inputs to
	/// @param Qs Q inputs, in a list captured by {}
	/// @param VKs VK inputs, in a list captured by {}
	CrossAttention(int num_heads, int qk_embed_size, int v_embed_size, int output_width, int input_width, std::initializer_list<Layer*> Qs={}, std::initializer_list<Layer*> VKs={}){
		init(num_heads, qk_embed_size, v_embed_size, output_width, input_width, Qs, VKs);
	}

	/// @brief Adds a layer as a VK input
	/// @param layer layer to add
	void add_VK(Layer* layer);

	/// @brief Adds a layer as a Q input
	/// @param layer layer to add
	void add_Q(Layer* layer);

	virtual void populate(float* params, float* gradients);
	virtual std::string get_type_name(){return "CrossAttention";}
private:
	// initialize layer
	void init(int num_heads_, int qk_embed_size_, int v_embed_size, int output_width, int input_width, std::initializer_list<Layer*> Qs, std::initializer_list<Layer*> VKs);

	// compute values for a single attention head
	void attention_head(float* Qin, float* VKin,
		float* qm, float* km, float* vm, float* zm,
		float* Q, float* K, float* KT, float* V, float* QKT,
		float* Z, float* O, const float norm_factor, bool calculateO);
	
	virtual void compute(float* input, float* output, float* intermediate_buffer, bool training);

	virtual bool compile_();

	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);
};

}

#endif