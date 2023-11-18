#ifndef CONV2D_HEADER
#define CONV2D_HEADER

#include "../activation_func.hpp"
#include "../layer.hpp"

namespace CPPML {

/* 
 * Passes a given number of 3d kernels over the input and 
 * outputs the result after passing through an activation function.
 * Takes in and puts out 2d or 3d vectors
 */
class Conv2d : public Layer {
public:
	// kernel size
	int kw, kh;

	// size of the padded input image, = input_shape + padding * 2;
	int pw, ph;

	// total size of 1 filter (kw * kh * input_shape.d)
	int filter_size;
	int padding;
	float *filters, *biases;
	float *filter_grads, *bias_grads;
	const ActivationFunc* activation;
	const bool use_bias;

	/// @param kw width of the kernel
	/// @param kh height of the kernel
	/// @param d depth of output
	/// @param activation activation to be applied to output
	/// @param padding amount of padding to add before convolving
	/// @param iw width to cast input to 
	/// @param ih height to cast input to
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Conv2d(int kw, int kh, int d, const ActivationFunc* const activation, int padding, int iw, int ih, Ts... input_layers) : Layer(input_layers...), use_bias(true){
		init(kw, kh, d, activation, padding, iw, ih);
	}

	/// @param kw width of the kernel
	/// @param kh height of the kernel
	/// @param d depth of output
	/// @param activation activation to be applied to output
	/// @param padding amount of padding to add before convolving
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Conv2d(int kw, int kh, int d, const ActivationFunc* const activation, int padding, Ts... input_layers) : Layer(input_layers...), use_bias(true){
		init(kw, kh, d, activation, padding, -1, -1);
	}

	/// @param kw width of the kernel
	/// @param kh height of the kernel
	/// @param d depth of output
	/// @param padding amount of padding to add before convolving
	/// @param use_bias whether to add bias after convolution or not
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Conv2d(int kw, int kh, int d, int padding, bool use_bias, Ts... input_layers) : Layer(input_layers...), use_bias(use_bias){
		init(kw, kh, d, nullptr, padding, -1, -1);
	}

	/// @param kw width of the kernel
	/// @param kh height of the kernel
	/// @param d depth of output
	/// @param activation activation to be applied to output
	/// @param padding amount of padding to add before convolving
	/// @param use_bias whether to add bias after convolution or not
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Conv2d(int kw, int kh, int d, const ActivationFunc* const activation, int padding, bool use_bias, Ts... input_layers) : Layer(input_layers...), use_bias(use_bias){
		init(kw, kh, d, activation, padding, -1, -1);
	}

	virtual void populate(float* params, float* gradients);

	virtual std::string get_type_name(){return "Conv2D";}

private:
	// initialize layer
	void init(int kw, int kh, int d, const ActivationFunc* const activation_, int padding, int iw, int ih);

	virtual void compute(float* input, float* output, float* intermediate_buffer, bool training);
	virtual bool compile_();
	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);

	// pads and image to the amount specified by this object
	float* pad_img(float* input, float* dest=nullptr);

	// takes in an image and flattens into rows of size
	// filter_size in the shape of the filter
	float* flatten_img(float* input, Shape in_shp, Shape out_shp, float* dst=nullptr);

	// handles calculating and adding the gradients for the current example
	void add_grads(float* input, float* out_change);
};

}

#endif