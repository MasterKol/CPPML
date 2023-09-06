#include "maxpooling2d.hpp"

#include <cfloat>
#include <iostream>

#include "../LinearAlgebra.hpp"

namespace CPPML {

void MaxPooling2d::init(int xScale_, int yScale_, int iw, int ih){
	assert(xScale_ > 0 && yScale_ > 0);
	xScale = xScale_;
	yScale = yScale_;
	input_shape = Shape(iw, ih, 0);
}

bool MaxPooling2d::compile_(){
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
			std::cerr << "CNN Dimentions do not match.\n\tExpected: (" << input_shape.w() << ", "
				<< input_shape.w() << ") got: (" << os.w() << ", " << os.h() << ")\n";
			exit(-1);
		}
		input_shape.d(input_shape.d() + os.size() / multiple);
	}

	// set output_shape
	output_shape = Shape((input_shape.w() + xScale - 1) / xScale, 
						 (input_shape.h() + xScale - 1) / yScale,
						 input_shape.d());

	intermediate_num = output_shape.size();
	num_params = 0;

	return false;
}

// no population needs to be done as this layer has no params
void MaxPooling2d::populate(float* params, float* gradients){}

inline void MaxPooling2d::process_in_line(float* inlayer, float* otlayer,
									   int* selected, int& oi, int& ii, int xhang){
	// loop over all output rows except last
	for(int ox = 0; ox < output_shape.w() - 1; ox++){
		// loop over all input rows in this output row
		for(int lx = 0; lx < xScale; lx++){
			// update output and selected if this input is larger
			if(otlayer[oi] < inlayer[ii]){
				otlayer[oi] = inlayer[ii];
				if(selected){selected[oi] = ii;}
			}
			ii++;
		}
		oi++;
	}

	// last column in output
	// loop over all input rows the last output row (xhang)	
	for(int lx = 0; lx < xhang; lx++){
		// update output and selected if this input is larger
		if(otlayer[oi] < inlayer[ii]){
			otlayer[oi] = inlayer[ii];
			if(selected){selected[oi] = ii;}
		}
		ii++;
	}
	oi++;
}

inline void MaxPooling2d::process_out_line(float* inlayer, float* otlayer,
								int* selected, int& oi, int& ii, int xhang, int maxY){
	int toi = oi;

	// loop over all input rows in an single output row
	for(int ly = 0; ly < maxY; ly++){
		oi = toi;
		process_in_line(inlayer, otlayer, selected, oi, ii, xhang);
	}
}

void MaxPooling2d::compute(float* input, float* output, float* intermediate_buffer){
	float* inlayer = input;
	float* otlayer = output;
	int* selected = (int*)intermediate_buffer;

	// size of the last input column
	const int xhang = ((input_shape.w() - 1) % xScale) + 1;
	// size of the last input row
	const int yhang = ((input_shape.h() - 1) % yScale) + 1;
	
	const float ninif = -FLT_MAX;
	const int otlayer_size = output_shape.w() * output_shape.h();

	for(int d = 0; d < input_shape.d(); d++){
		// fill output with -infinity so that max of initial and
		// first value just returns the first value
		vDSP_vfill(&ninif, otlayer, 1, otlayer_size);

		// selected does not need to be zeroed as the first input will
		// always overwrite the default value of otlayer

		// loop over all output rows except last
		int oi = 0, ii = 0; // oi = output index, ii = input index
		for(int oy = 0; oy < output_shape.h() - 1; oy++){
			process_out_line(inlayer, otlayer, selected, oi, ii, xhang, yScale);
		}

		// last row in input
		process_out_line(inlayer, otlayer, selected, oi, ii, xhang, yhang);

		// move to next layer
		inlayer  += input_shape.w() * input_shape.h();
		otlayer  += otlayer_size;
		if(selected){selected += otlayer_size;}
	}
}

void MaxPooling2d::get_change_grads(float* out_change, float* inpt_change, float* input, float* output, float* intermediate){
	float* inlayer = inpt_change;
	float* otlayer = out_change;
	int* selected = (int*)intermediate;

	// size of one 'slice' of output
	const int olsize = output_shape.w() * output_shape.h();
	const int ilsize = input_shape.w() * input_shape.h();

	for(int d = 0; d < output_shape.d(); d++){
		// move out change to the place that it came from in input
		// which was stored in selected
		for(int i = 0; i < olsize; i++){
			inlayer[selected[i]] = otlayer[i];
		}

		// move to next layer
		inlayer  += ilsize;
		otlayer  += olsize;
		selected += olsize;
	}
}

}