#ifndef MAXPOOLING2D_HEADER
#define MAXPOOLING2D_HEADER

#include "../layer.hpp"

/*
 * Takes in a set of 2d matrices and scales it down by the given
 * factor. Captures the max of each region in the input.
 */
class MaxPooling2d : public Layer {
public:
	int xScale, yScale;

	template<typename... Ts>
	MaxPooling2d(int xScale, int yScale, int iw, int ih, Ts... input_layers) : Layer(input_layers...){
		init(xScale, yScale, iw, ih);
	}

	template<typename... Ts>
	MaxPooling2d(int xScale, int yScale, Ts... input_layers) : Layer(input_layers...){
		init(xScale, yScale, -1, -1);
	}

	template<typename... Ts>
	MaxPooling2d(int factor, Ts... input_layers) : Layer(input_layers...){
		init(factor, factor, -1, -1);
	}

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "MaxPooling";}

private:
	// initialize layer
	void init(int xScale, int yScale, int iw, int ih);

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

	// processes a single line of input and writes to given output
	inline void process_in_line(float* inlayer, float* otlayer,
							int* selected, int& oi, int& ii, int xhang);

	// process a single of line of output
	inline void process_out_line(float* inlayer, float* otlayer,
							int* selected, int& oi, int& ii, int xhang, int maxY);
};

#endif