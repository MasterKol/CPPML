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

	/// @param xScale up-scaling factor in width
	/// @param yScale up-scaling factor in height
	/// @param xPadding padding to add in width after upscaling, left biased
	/// @param yPadding padding to add in height after upscaling, top biased
	/// @param iw width to cast input to 
	/// @param ih height to cast input to
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Upscale2d(int xScale, int yScale, int xPadding, int yPadding, int iw, int ih, Ts... input_layers) : Layer(input_layers...){
		init(xScale, yScale, xPadding, yPadding, iw, ih);
	}

	/// @param xScale up-scaling factor in width
	/// @param yScale up-scaling factor in height
	/// @param xPadding padding to add in width after upscaling, left biased
	/// @param yPadding padding to add in height after upscaling, top biased
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Upscale2d(int xScale, int yScale, int xPadding, int yPadding, Ts... input_layers) : Layer(input_layers...){
		init(xScale, yScale, xPadding, yPadding, -1, -1);
	}

	/// @param factor up-scaling factor in width and height
	/// @param padding padding to add in width and height after upscaling, top-left biased
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	Upscale2d(int factor, int padding, Ts... input_layers) : Layer(input_layers...){
		init(factor, factor, padding, padding, -1, -1);
	}

	virtual void populate(float* params, float* gradients);

	virtual std::string get_type_name(){return "Upscale2D";}

private:
	// initialize layer
	void init(int xScale, int yScale, int xPadding, int yPadding, int iw, int ih);

	virtual void compute(float* input, float* output, float* intermediate_buffer, bool training);

	virtual bool compile_();

	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);
};

}

#endif