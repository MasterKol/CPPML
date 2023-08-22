#ifndef CONV2D_HEADER
#define CONV2D_HEADER

#include "../activation.hpp"
#include "../layer.hpp"

namespace CPPML {

/* 
 * Passes a given number of 3d kernels over the input and 
 * outputs the result after passing through an activation function.
 * Takes in and puts out 2d or 3d vectors
 */
class Conv2d : public Layer {
public:
	int kw, kh;
	int pw, ph;
	int filter_size;
	int padding;
	float *filters, *biases;
	float *filter_grads, *bias_grads;
	const Activation* activation;

	template<typename... Ts>
	Conv2d(int kw, int kh, int d, const Activation* const activation_, int padding, int iw, int ih, Ts... input_layers) : Layer(input_layers...){
		init(kw, kh, d, activation_, padding, iw, ih);
	}

	template<typename... Ts>
	Conv2d(int kw, int kh, int d, const Activation* const activation_, int padding, Ts... input_layers) : Layer(input_layers...){
		init(kw, kh, d, activation_, padding, -1, -1);
	}

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "Conv2D";}

private:
	// initialize layer
	void init(int kw, int kh, int d, const Activation* const activation_, int padding, int iw, int ih);

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

	// pads and image to the amount specified by this object
	float* pad_img(float* input, float* dest = NULL);

	// takes in an image and flattens into rows of size
	// filter_size in the shape of the filter
	float* flatten_img(float* input, Shape in_shp, Shape out_shp, float* dst=NULL);

	// handles calculating and adding the gradients for the current example
	void add_grads(float* input, float* out_change);
};

}

#endif