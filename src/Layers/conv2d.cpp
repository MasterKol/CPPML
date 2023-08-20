#include "conv2d.hpp"
#include "../activation.hpp"
#include "../helper.hpp"

#include "../LinearAlgebra.hpp"
#include <assert.h>

namespace CPPML {

void Conv2d::init(int kw_, int kh_, int d_,
				  const Activation* const activation_,
				  int padding_, int iw, int ih){
	assert(kw_ > 0 && kh_ > 0 && d_ > 0 && padding_ >= 0);
	kw = kw_;
	kh = kw_;
	padding = padding_;
	output_shape.d = d_;
	input_shape = Shape(iw, ih, 0);
	activation = activation_;
}

bool Conv2d::compile_(){
	// if input shape was set to auto then set to first input shape
	if(input_shape.w == -1){
		input_shape = inputs[0]->output_shape;
		input_shape.d = 0; // set to zero because it will be re added
	}

	// size of one input slice, every input must be a multiple of this
	const int multiple = input_shape.w * input_shape.h;

	// loop over inputs and generate input shape
	for(Layer* l : inputs){
		Shape os = l->output_shape;
		// check if inputs match in the correct dimensions
		// if the input is flat try to fix it else throw error
		if(!((os.d == 1 && os.h == 1 && os.w % multiple == 0) ||
		   (os.h != 1 && os.w == input_shape.w && os.h == input_shape.h))){
			printf("expected: (%d, %d); got: (%d, %d)\n", input_shape.w, input_shape.h, os.w, os.h);
			fflush(stdout);
			throw std::runtime_error("CNN dimensions do not match");
		}
		input_shape.d += os.size / multiple;
	}
	// fix size of input
	input_shape.fix_size();

	// set output_shape
	output_shape = Shape(input_shape.w - kw + 1 + 2 * padding,
						 input_shape.h - kh + 1 + 2 * padding,
						 output_shape.d);

	filter_size = kw * kh * input_shape.d;
	// there are 'depth' filters and one bias for each output
	num_params = (filter_size + 1) * output_shape.d;
	intermediate_num = output_shape.size;

	// size of the padded input image
	pw = input_shape.w + padding * 2;
	ph = input_shape.h + padding * 2;

	return false;
}

void Conv2d::populate(float* params, float* gradients){
	filters = params + output_shape.d;
	biases = params;

	filter_grads = gradients + output_shape.d;
	bias_grads = gradients;

	// apparently this is how you are supposed to
	// initialize the filters in a conv2d layer
	const float sdv = sqrtf(2.0f / filter_size);
	for(int i = 0; i < filter_size * output_shape.d; i++){
		filters[i] = randomGaussian(0, sdv);
	}
}

float* Conv2d::pad_img(float* input, float* dest){
	// create buffer for storing padded image
	float* padded = dest;
	if(dest == NULL){
		padded = new float[pw * ph * input_shape.d];
	}

	float* img = padded; // current part of output that is being modified
	float* rp = input;	 // current part of input  that is being read
	// loop over all input 'slices'
	for(int d = 0; d < input_shape.d; d++){
		// fill top `padding` rows of 'slice' with zeros
		memset(img, 0, pw * padding * sizeof(float));
		img += pw * padding;
		
		// loop over all rows in input image
		for(int y = 0; y < input_shape.h; y++){
			// zero first `padding` columns of row
			memset(img, 0, padding * sizeof(float));

			// copy over data from input
			memcpy(img + padding, rp, input_shape.w * sizeof(float));

			// zero last `padding` columns of row
			memset(img + pw - padding, 0, padding * sizeof(float));
			rp += input_shape.w;
			img += pw;
		}

		// fill bottom `padding` rows of 'slice' with zeros
		memset(img, 0, pw * padding * sizeof(float));
		img += pw * padding;
	}

	return padded;
}

void Conv2d::compute(float* input, float* output, float* intermediate_buffer){
	// if image needs to be padded than it will be stored here
	// otherwise this is just and alias for input
	float* padded = input;
	if(padding != 0){ // pad if nessisary
		padded = pad_img(input);
	}

	// size of one slice of the output image
	const int output_size = output_shape.w * output_shape.h;

	// turn padded image into matrix form
	float* img_mat = flatten_img(padded, Shape(pw, ph, input_shape.d), output_shape);

	if(padding != 0){ // free padded img if necessary
		delete[] padded;
	}
	padded = NULL;

	// place to write value of convolution before activation
	// fuction. if there is no intermediate buffer just write
	// to output as intermediate
	float* inter_s = intermediate_buffer;
	if(inter_s == NULL){
		inter_s = output;
	}

	// pointer to current output 'slice'
	float* out_s = output;
	// loop over all output 'slices'
	
	for(int d = 0; d < output_shape.d; d++){
		// perform matrix mult that is equivelent to the convolution
		vDSP_mmul(img_mat, 1, filters + filter_size * d,
					1, inter_s, 1, output_size, 1, filter_size);
		
		// add bias
		vDSP_vsadd(inter_s, 1, biases + d, inter_s, 1, output_size);

		// perform activation on output
		activation->f(inter_s, out_s, output_size);

		// add output_size to move to next 'slice'
		inter_s += output_size;
		out_s += output_size;
	}

	// free the matrix used for storing flattened image
	delete[] img_mat;
}

void Conv2d::add_grads(float* input, float* out_change){
	// pad input if necessary
	float* in_padded = input;
	if(padding != 0){
		in_padded = pad_img(input);
	}

	// size of output slice
	const int output_slice = output_shape.w * output_shape.h;
	// size of one slice of padded input
	const int pd_size = pw * ph;
	//int mat_size = std::max(, input_shape.w * input_shape.h * output_shape.d * kw * kh);
	float* in_img_mat = new float[output_slice * filter_size]; // FIXME

	float* in_slice = in_padded;
	float* mat_row = in_img_mat;
	// loop over all positions in the filter
	for(int d = 0; d < input_shape.d; d++){
		for(int y = 0; y < kh; y++){
			for(int x = 0; x < kw; x++){
				// move sub matrix of input to img_mat
				// because of the way the matrix is in memory it gets flattened
				// into a row vector for free
				vDSP_mmov(in_slice + y * pw + x, mat_row, output_shape.w, output_shape.h, pw, output_shape.w);
				mat_row += output_slice;
			}
		}

		in_slice += pd_size;
	}

	// free padded if necessary
	if(padding != 0){
		delete[] in_padded;
	}

	float* out_grad_slice = out_change;
	float* cur_filter_grads = filter_grads;
	float* t_filter_grads = new float[filter_size]; // assign memory because it may be too big for the stack

	// the below code modifies the gradients so guard it with a mutex
	std::lock_guard<std::mutex> guard(gradient_mutex);

	// loop over all of the filters
	for(int d = 0; d < output_shape.d; d++){
		// get gradients for filter and write to temp var
		vDSP_mmul(in_img_mat, 1, out_grad_slice, 1, t_filter_grads, 1, filter_size, 1, output_slice);

		vDSP_vadd(cur_filter_grads, 1, t_filter_grads, 1, cur_filter_grads, 1, filter_size);

		out_grad_slice += output_slice;
		cur_filter_grads += filter_size;
	}

	const int owh = output_shape.w * output_shape.h;
	for(int i = 0; i < output_shape.d; i++){
		// total vector and write to bias grad
		float t;
		vDSP_sve(out_change + owh * i, 1, &t, owh);
		bias_grads[i] += t;
	}

	delete[] t_filter_grads;
	delete[] in_img_mat;
}

void Conv2d::get_change_grads(float* out_change, float* inpt_change,
					float* input, float* output, float* intermediate){
	// apply derivative of activation function to intermediate
	activation->df(intermediate, intermediate, output_shape.size);
	
	// multiply intermediate by change from previous layer
	vDSP_vmul(intermediate, 1, out_change, 1, out_change, 1, output_shape.size);

	const int pkw = kw - 1 - padding; // padding to add
	const int pkh = kh - 1 - padding;
	const int iw  = output_shape.w + pkw*2; // pad change
	const int ih  = output_shape.h + pkh*2;

	// out_change 0 padded to be of size iw x ih 
	float* padded = new float[iw * ih * output_shape.d];
	
	// pad gradients from output to iw x ih
	// I don't use the pad method as this is sufficiently different
	for(int d = 0; d < output_shape.d; d++){
		memset(padded + d * ih * iw, 0, sizeof(float) * iw * pkh);
		for(int y = 0; y < output_shape.h; y++){
			float* const padded_row = padded + (d * ih + y + pkh) * iw;
			memset(padded_row, 0, sizeof(float) * pkw);
			memset(padded_row + output_shape.w + pkw, 0, sizeof(float) * pkw);

			memcpy(padded_row + pkh, out_change + (d * output_shape.h + y) * output_shape.w, sizeof(float) * output_shape.w);
		}
		memset(padded + (d * ih + ih - pkh) * iw, 0, sizeof(float) * iw * pkh);
	}

	// flatten padded image to matrix form
	float* img_mat = flatten_img(padded, Shape(iw, ih, output_shape.d), input_shape);

	// free padded image as it's no longer needed
	delete[] padded;

	// useful constants for later
	const int fs = kw * kh; // size of one 'slice' of a kernel
	const int block_length = fs * output_shape.d; // width of img_mat
	// size of out input slice
	const int input_slice = input_shape.w * input_shape.h;
	
	// temp storage for transposed kernels/filters
	float* filter = new float[block_length];
	for(int f = 0; f < input_shape.d; f++){
		// flip the kernel along x, y, and d
		for(int d = 0; d < output_shape.d; d++){
			float* dst = filter + d * fs + fs - 1;
			float* src = filters + d * filter_size + f * fs;
			for(int i = 0; i < fs; i++){
				*(dst - i) = *(src + i);
			}
		}

		// perform matrix multiplication between img_mat and transformed filter
		vDSP_mmul(img_mat, 1, filter, 1, inpt_change + f * input_slice, 1, input_slice, 1, block_length);
	}
	delete[] filter;
	delete[] img_mat;

	add_grads(input, out_change);
}

float* Conv2d::flatten_img(float* input, Shape in_shp, Shape out_shp, float* dst){
	// matrix version of the image. Maps the image to a matrix
	// of size filtersize x output_size. Each row represents one
	// output pixel (in order). This allows the convolution to be
	// carried out as a simple matrix multiplication
	const int block_size = kw * kh * in_shp.d;
	float* img_mat = dst;
	if(dst == NULL){
		img_mat = new float[block_size * out_shp.w * out_shp.h];
	}

	// code to flatten matrix
	// the loops are in this order because it's what I've found
	// to run fastest, not really sure why. Probably has to do
	// with the cache not being 'trashed'
	for(int y = 0; y < out_shp.h; y++){
		for(int x = 0; x < out_shp.w; x++){
			int woff = (y * out_shp.w + x) * block_size;
			for(int j = 0; j < kh; j++){
				for(int d = 0; d < in_shp.d; d++){
					memcpy(img_mat + woff + (d * kh + j) * kw, input + (d * in_shp.h + y + j) * in_shp.w + x, kw * sizeof(float));
				}
			}
		}
	}

	return img_mat;
}

}