#ifndef UPSCALE2D_HEADER
#define UPSCALE2D_HEADER

#include "../layer.hpp"

namespace CPPML {

/*
 * Scales a 2d layer up by the factor given. Fills empty space with zeros.
 * Additionally padding can be added, this adds a TOTAL of the given padding
 * to the top/bottom and left/right allowing for odd padding. Numbers are put
 * into the center (left justified) of their output region and the padding is
 * added first to the top then the bottom.
 */
class Upscale2d : public Layer {
public:
	int xScale, yScale;
	int xPadding, yPadding;

	template<typename... Ts>
	Upscale2d(int xScale, int yScale, int xPadding, int yPadding, int iw, int ih, Ts... input_layers) : Layer(input_layers...){
		init(xScale, yScale, xPadding, yPadding, iw, ih);
	}

	template<typename... Ts>
	Upscale2d(int xScale, int yScale, int xPadding, int yPadding, Ts... input_layers) : Layer(input_layers...){
		init(xScale, yScale, xPadding, yPadding, -1, -1);
	}

	template<typename... Ts>
	Upscale2d(int factor, int padding, Ts... input_layers) : Layer(input_layers...){
		init(factor, factor, padding, padding, -1, -1);
	}

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "Upscale2D";}

private:
	// initialize layer
	void init(int xScale, int yScale, int xPadding, int yPadding, int iw, int ih);

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