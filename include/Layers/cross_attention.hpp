#ifndef CROSS_ATTENTION_HEADER
#define CROSS_ATTENTION_HEADER

#include <vector>

#include "../layer.hpp"
#include <initializer_list>

namespace CPPML {

/*
 * DON'T USE add_input FOR THIS LAYER, USE add_VK / add_Q
 * Performs multi-head cross attention on two different sets
 * of inputs. Outputs with specified width and height of Q inputs
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

	CrossAttention(int num_heads, int qk_embed_size, int v_embed_size, std::initializer_list<Layer*> Qs={}, std::initializer_list<Layer*> VKs={}){
		init(num_heads, qk_embed_size, v_embed_size, -1, -1, -1, Qs, VKs);
	}

	CrossAttention(int num_heads, int qk_embed_size, int v_embed_size, int output_width, std::initializer_list<Layer*> Qs={}, std::initializer_list<Layer*> VKs={}){
		init(num_heads, qk_embed_size, v_embed_size, output_width, -1, -1, Qs, VKs);
	}

	CrossAttention(int num_heads, int qk_embed_size, int v_embed_size, int output_width, int Qwidth, int VKwidth, std::initializer_list<Layer*> Qs={}, std::initializer_list<Layer*> VKs={}){
		init(num_heads, qk_embed_size, v_embed_size, output_width, Qwidth, VKwidth, Qs, VKs);
	}

	// add a new layer 
	void add_VK(Layer* l);

	void add_Q(Layer* l);

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "CrossAttention";}

private:
	// initialize layer
	void init(int num_heads_, int qk_embed_size_, int v_embed_size, int output_width, int Qwidth, int VKwidth, std::initializer_list<Layer*> Qs, std::initializer_list<Layer*> VKs);

	// compute values for a single attention head
	void attention_head(float* Qin, float* VKin,
		float* qm, float* km, float* vm, float* zm,
		float* Q, float* K, float* KT, float* V, float* QKT,
		float* Z, float* O, const float norm_factor, bool calculateO);

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