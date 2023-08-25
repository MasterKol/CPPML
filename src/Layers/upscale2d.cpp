#include "upscale2d.hpp"

// left/top padding = (xyPadding + 1) / 2
// right/bottom padding = xyPadding / 2
// pos in region = (xyScale - 1) / 2

namespace CPPML {

void Upscale2d::init(int xScale_, int yScale_, int xPadding_, int yPadding_, int iw, int ih){
	assert(xScale_ > 0 && yScale_ > 0);
	xScale = xScale_;
	yScale = yScale_;
	xPadding = xPadding_;
	yPadding = yPadding_;
	input_shape = Shape(iw, ih, 0);
}

bool Upscale2d::compile_(){
	// if input shape was set to auto then set to first input shape
	// if input shape was set to auto then set to first input shape
	if(input_shape.w() == -1){
		input_shape = inputs[0]->output_shape;
		input_shape.d(0); // set to zero because it will be re added
	}

	// size of one input slice, every input must be a multiple of this
	const int multiple = input_shape.w() * input_shape.h();

	// loop over inputs and generate input shape
	for(Layer* l : inputs){
		Shape os = l->output_shape;
		// check if inputs match in the correct dimensions
		// if the input is flat try to fix it else throw error
		if(!((os.d() == 1 && os.h() == 1 && os.w() % multiple == 0) ||
		   (os.h() != 1 && os.w() == input_shape.w() && os.h() == input_shape.h()))){
			// FIXME print better error message
			throw std::runtime_error("Input dimensions do not match");
		}
		input_shape.d(input_shape.d() + os.size() / multiple);
	}

	// set output_shape
	output_shape = Shape(input_shape.w() * xScale + xPadding,
						 input_shape.h() * yScale + yPadding,
						 input_shape.d());
	
	// intermediate_num is 0
	// param_num is 0

	return false;
}

// no population needs to be done as this layer has no params
void Upscale2d::populate(float* params, float* gradients){}

void Upscale2d::compute(float* input, float* output, float* intermediate_buffer){
	// pointers to current position in input/output
	float* out_cur = output;
	float* in_cur  = input;
	
	// size of one output layer
	const int ot_size = output_shape.w() * output_shape.h();

	// amount to move from the end of one output row to the start of the next
	const int inter_row = xPadding + (yScale - 1) * output_shape.w();
	
	// set the initial offset of the output pointer
	out_cur += ((yPadding + 1)/2 + (yScale - 1)/2) * output_shape.w() + ((xPadding + 1)/2 + (xScale - 1)/2);

	// loop over all layers
	for(int d = 0; d < input_shape.d(); d++){
		// zero this layer
		memset(output + ot_size * d, 0, ot_size * sizeof(float));

		// loop over all input rows
		for(int y = 0; y < input_shape.h(); y++){
			// loop over all input columns
			for(int x = 0; x < input_shape.w(); x++){
				*out_cur = *in_cur;
				in_cur++;
				out_cur += xScale;
			}
			// skip over output padding
			out_cur += inter_row;
		}

		// skip to next layer
		out_cur += yPadding * output_shape.w();
	}
}

// This is exactly the same as the 
void Upscale2d::get_change_grads(float* out_change, float* inpt_change, float* input, float* output, float* intermediate){
	// pointers to current position in input/output
	float* out_cur = out_change;
	float* in_cur  = inpt_change;

	// amount to move from the end of one output row to the start of the next
	const int inter_row = xPadding + (yScale - 1) * output_shape.w();
	
	// set the initial offset of the output pointer
	out_cur += ((yPadding + 1)/2 + (yScale - 1)/2) * output_shape.w() + ((xPadding + 1)/2 + (xScale - 1)/2);

	// loop over all layers
	for(int d = 0; d < input_shape.d(); d++){
		// zero this layer
		//memset(in_cur, 0, in_size * sizeof(float));

		// loop over all input rows
		for(int y = 0; y < input_shape.h(); y++){
			// loop over all input columns
			for(int x = 0; x < input_shape.w(); x++){
				*in_cur = *out_cur;
				in_cur++;
				out_cur += xScale;
			}
			// skip over output padding
			out_cur += inter_row;
		}

		// skip to next layer
		out_cur += yPadding * output_shape.w();
	}
}

}