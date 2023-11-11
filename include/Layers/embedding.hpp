#ifndef EMBEDDING_LAYER_HEADER
#define EMBEDDING_LAYER_HEADER

#include "../layer.hpp"

namespace CPPML {

/*
 * Creates a trainable embeddings for discrete inputs,
 * input layer must have only 1 integer value in the range
 * [0, num_classes-1].
 * This layer cannot send gradients backwards so it should
 * come after an input layer.
 */
class Embedding : public Layer {
public:
	const int num_classes;
	const int embedding_length;

	float *params, *gradients;

	/// @param num_classes number of classes to learn
	/// @param embedding_length size of each class embedding
	/// @param input_layer
	Embedding(int num_classes, int embedding_length, Layer* input_layer=nullptr);

	virtual void populate(float* params, float* gradients);

	virtual std::string get_type_name(){return "Embedding";}
private:
	virtual void compute(float* input, float* output, float* intermediate_buffer, bool training);

	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);

	virtual bool compile_();
};


} // namespace CPPML

#endif