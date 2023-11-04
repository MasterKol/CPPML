#ifndef MAXPOOLING2D_HEADER
#define MAXPOOLING2D_HEADER

#include "../layer.hpp"

namespace CPPML {

/*
 * Takes in a set of 2d matrices and scales it down by the given
 * factor. Captures the max of each region in the input.
 */
class MaxPooling2d : public Layer {
public:
	int xScale, yScale;

	/// @param xScale down-scaling factor in width
	/// @param yScale down-scaling factor in height
	/// @param iw width to cast input to 
	/// @param ih height to cast input to
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	MaxPooling2d(int xScale, int yScale, int iw, int ih, Ts... input_layers) : Layer(input_layers...){
		init(xScale, yScale, iw, ih);
	}

	/// @param xScale down-scaling factor in width
	/// @param yScale down-scaling factor in height
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	MaxPooling2d(int xScale, int yScale, Ts... input_layers) : Layer(input_layers...){
		init(xScale, yScale, -1, -1);
	}

	/// @param factor down-scaling factor in width and height
	/// @param input_layers vararg, inputs to this layer
	template<typename... Ts>
	MaxPooling2d(int factor, Ts... input_layers) : Layer(input_layers...){
		init(factor, factor, -1, -1);
	}

	virtual void populate(float* params, float* gradients);
	virtual std::string get_type_name(){return "MaxPooling";}

private:
	// initialize layer
	void init(int xScale, int yScale, int iw, int ih);

	virtual void compute(float* input, float* output, float* intermediate_buffer, bool training);
	virtual bool compile_();
	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);

	// processes a single line of input and writes to given output
	inline void process_in_line(float* inlayer, float* otlayer,
							int* selected, int& oi, int& ii, int xhang);

	// process a single of line of output
	inline void process_out_line(float* inlayer, float* otlayer,
							int* selected, int& oi, int& ii, int xhang, int maxY);
};

}

#endif