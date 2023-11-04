#include "image_flatten.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "../LinearAlgebra.hpp"

namespace CPPML {

void create_emb_list(float* const emb_start, const int dim, const int num_embs, const int outWidth);
void to_image(float* mat, float* img, Shape mat_shape, Shape img_shape, int xPatchSize, int yPatchSize, int xPatches, int yPatches);
void to_matrix(float* mat, float* img, Shape mat_shape, Shape img_shape, int xPatchSize, int yPatchSize, int xPatches, int yPatches);

void ImageFlatten::Init(int xPatchSize_, int yPatchSize_, int xEmbSize_, int yEmbSize_, Layer* l){
	xPatchSize = xPatchSize_;
	yPatchSize = yPatchSize_;
	xEmbSize = xEmbSize_;
	yEmbSize = xEmbSize_;

	assert(xPatchSize > 0 && yPatchSize > 0);
	assert(xEmbSize >= 0 && yEmbSize >= 0);
	assert(xEmbSize % 2 == 0 && yEmbSize % 2 == 0); // embeds must be even

	img_in = l;
	add_input(l);

	// make last element of inputs the first (make image come first)
	std::rotate(inputs.begin(), inputs.end() - 1, inputs.end());
}

bool ImageFlatten::compile_(){
	image_shape = img_in->output_shape;

	int size = 0;
	for(Layer* l : inputs){
		size += l->output_shape.size();
	}
	input_shape = Shape(size);

	xPatches = (image_shape.w() + xPatchSize - 1) / xPatchSize;
	yPatches = (image_shape.h() + yPatchSize - 1) / yPatchSize;

	extra_embed_size = size - image_shape.size();

	output_shape = Shape(xPatchSize * yPatchSize * image_shape.d() + xEmbSize + yEmbSize + extra_embed_size, xPatches * yPatches);

	intermediate_num = 0;
	num_params = 0;
	return false;
}

void ImageFlatten::populate(float* params, float* gradients){}

void ImageFlatten::compute(float* input, float* output, float* intermediate_buffer, bool training){
	to_matrix(output, input, output_shape, image_shape, xPatchSize, yPatchSize, xPatches, yPatches);

	// --==== add position embeddings ====--

	// create x embeds for first xPatches rows
	float* xemb_start = output + xPatchSize * yPatchSize * image_shape.d();
	create_emb_list(xemb_start, xEmbSize, xPatches, output_shape.w());

	// copy x embeds from first set of rows all other rows
	for(int i = 1; i < yPatches; i++){
		vDSP_mmov(xemb_start, xemb_start + output_shape.w() * xPatches * i, xEmbSize, xPatches, output_shape.w(), output_shape.w());
	}

	// create y embeds for first yPatches rows
	float* const yemb_start = output + xPatchSize * yPatchSize * image_shape.d() + xEmbSize;
	create_emb_list(yemb_start, yEmbSize, yPatches, output_shape.w());

	// copy y embeds from first set of rows all other rows
	// start at end, fill last xPatches rows with last row yPatches-1, etc
	for(int i = yPatches - 1; i >= 0; i--){
		for(int j = 0; j < xPatches; j++){
			memcpy(yemb_start + (i * xPatches + j) * output_shape.w(), yemb_start + i * output_shape.w(), yEmbSize * sizeof(float));
		}
	}

	// --==== add other embeddings ====--

	if(input_shape.size() != image_shape.size()){
		float* embed = input + image_shape.size();
		float* embed_start = output + xPatchSize * yPatchSize * image_shape.d() + xEmbSize + yEmbSize;

		for(int i = 0; i < output_shape.h(); i++){
			memcpy(embed_start + i * output_shape.w(), embed, extra_embed_size * sizeof(float));
		}
	}
}

// creates sinusoidal position embedding of given dimension
// writing sines and cosines to given arrays
void fill_emb_start(float* sins, float* coss, int dim){
	const float W = powf(30.0f * (float)dim - 50.0f, -2.0f / (float)dim);

	float wk = W;
	for(int k = 0; k < dim / 2; k++){
		sins[k] = sinf(wk);
		coss[k] = cosf(wk);
		wk *= W;
	}
}

// more efficient version of fill_emb_start
// fills the first 'dim' columns of an 'num_embs' x 'outWidth' matrix
// with sin. pos. embedding where embedding matches the row
void create_emb_list(float* const emb_start, const int dim, const int num_embs, const int outWidth){
	if(dim == 0)
		return;
	const int half_dim = dim / 2;

	float* const sins = emb_start;
	float* const coss = emb_start + half_dim;

	// create embedding vectors for x = 0
	fill_emb_start(sins, coss, dim);
	float* sinSrc = sins;
	float* cosSrc = coss;
	// loop over all rows, first row was set by fill_emb_start
	for(int i = 1; i < num_embs; i++){
		float* const sinDst = sinSrc + outWidth;
		float* const cosDst = cosSrc + outWidth;

		// cos(W(x+1)) = cos(W*1)*cos(Wx) - sin(W*1)*sin(Wx)
		vDSP_vmmsb(cosSrc, 1, coss, 1, sinSrc, 1, sins, 1, cosDst, 1, half_dim);

		// sin(W(x+1)) = sin(W*1)*cos(Wx) + cos(W*1)*sin(Wx)
		vDSP_vmul(sinSrc, 1, coss, 1, sinDst, 1, half_dim); // sinDst <- sin(W*1)*cos(Wx)
		vDSP_vma(cosSrc, 1, sins, 1, sinDst, 1, sinDst, 1, half_dim); // sinDst += cos(W*1)*sin(Wx)

		// move to next row
		sinSrc = sinDst;
		cosSrc = cosDst;
	}
}

void ImageFlatten::get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate){
	to_image(out_change, inpt_change, output_shape, image_shape, xPatchSize, yPatchSize, xPatches, yPatches);

	// get embed gradients
	float* emb_grad = inpt_change + image_shape.size();
	memset(emb_grad, 0, extra_embed_size * sizeof(float));

	float* grad_out = out_change + output_shape.w() - extra_embed_size;
	for(int i = 0; i < output_shape.h(); i++){
		vDSP_vadd(emb_grad, 1, grad_out + output_shape.w() * i, 1, emb_grad, 1, extra_embed_size);
	}
}

/*-------======== IMAGE DE-FLATTEN LAYER ========------- */

ImageDeFlatten::ImageDeFlatten(int xPatchSize, int yPatchSize, int imgw, int imgh, int imgd, Layer* l){
	flatten_layer = nullptr;
	init(xPatchSize, yPatchSize, imgw, imgh, imgd, l);
}

ImageDeFlatten::ImageDeFlatten(ImageFlatten* flatten_layer_, Layer* l){
	flatten_layer = flatten_layer_;
	init(flatten_layer->xPatchSize, flatten_layer->yPatchSize, 
			-1, -1, -1, l);
	
}

void ImageDeFlatten::init(int xPatchSize_, int yPatchSize_, int imgw, int imgh, int imgd, Layer* l){
	xPatchSize = xPatchSize_;
	yPatchSize = yPatchSize_;

	output_shape = Shape(imgw, imgh, imgd);

	if(l)
		add_input(l);

	intermediate_num = 0;
	num_params = 0;
}

void ImageDeFlatten::populate(float* params, float* gradients){}

bool ImageDeFlatten::compile_(){
	assert(inputs.size() == 1 && "DeFlatten layer must have exactly 1 input");

	if(flatten_layer){
		output_shape = flatten_layer->img_in->output_shape;
	}

	xPatches = (output_shape.w() + xPatchSize - 1) / xPatchSize;
	yPatches = (output_shape.h() + yPatchSize - 1) / yPatchSize;

	input_shape = inputs[0]->output_shape;//Shape(xPatchSize * yPatchSize, xPatches * yPatches);

	const int expected_input_len = xPatchSize * yPatchSize * xPatches * yPatches * output_shape.d();
	if(input_shape.size() != expected_input_len){
		std::cerr << "Error in Image Deflatten layer:\n";
		std::cerr << "    Input length does not match expected length. Got: " << input_shape.size() << ", Expected: " << expected_input_len << std::endl;
		std::cerr << "    Lengths must match, Input shape: " << input_shape.to_string() << ", expected: " << Shape(xPatchSize * yPatchSize * output_shape.d(), xPatches * yPatches).to_string() << std::endl;
		exit(1);
	}

	//input_shape.d(inputs[0]->output_shape.d());

	return false;
}

void ImageDeFlatten::compute(float* input, float* output, float* intermediate_buffer, bool training){
	to_image(input, output, input_shape, output_shape, xPatchSize, yPatchSize, xPatches, yPatches);
}

void ImageDeFlatten::get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate){
	to_matrix(inpt_change, out_change, input_shape, output_shape, xPatchSize, yPatchSize, xPatches, yPatches);
}

// takes in an image that has been flattened to a
// matrix and reforms it back into an image
void to_image(float* mat, float* img, Shape mat_shape, Shape img_shape, int xPatchSize, int yPatchSize, int xPatches, int yPatches){
	// amount that x and y overhang by
	const int x_hang = ((img_shape.w() - 1) % xPatchSize) + 1;
	const int y_hang = ((img_shape.h() - 1) % yPatchSize) + 1;

	const int img_slice = img_shape.w() * img_shape.h();
	const int patch_size = xPatchSize * yPatchSize;

	// loop over all rows that 100% don't have 'overhang'
	for(int y = 0; y < yPatches-1; y++){
		// loop over all columns that 100% don't have 'overhang'
		for(int x = 0; x < xPatches - 1; x++){
			// move row of matrix into correct spot on image
			for(int d = 0; d < img_shape.d(); d++)
				vDSP_mmov(mat + patch_size * d, img + img_slice * d, xPatchSize, yPatchSize, xPatchSize, img_shape.w());
			// go to next row and next image patch
			mat += mat_shape.w();
			img += xPatchSize;
		}

		// move last column, might be overhang so use x_hang instead of xPatchSize
		for(int d = 0; d < img_shape.d(); d++)
			vDSP_mmov(mat + patch_size * d, img + img_slice * d, x_hang, yPatchSize, xPatchSize, img_shape.w());
		mat += mat_shape.w();

		// move to next row of output image
		img += img_shape.w() * (yPatchSize - 1) + x_hang;
	}

	// process all of last row separately as it might have 'overhang' 
	for(int x = 0; x < xPatches - 1; x++){
		for(int d = 0; d < img_shape.d(); d++)
			vDSP_mmov(mat + patch_size * d, img + img_slice * d, xPatchSize, y_hang, xPatchSize, img_shape.w());
		mat += mat_shape.w();
		img += xPatchSize;
	}

	// move bottom right corner, may have x and y overhang so this is an extra special case
	for(int d = 0; d < img_shape.d(); d++)
			vDSP_mmov(mat + patch_size * d, img + img_slice * d, x_hang, y_hang, xPatchSize, img_shape.w());
}

// takes in an image and flattens it into a matrix of the given shape
void to_matrix(float* mat, float* img, Shape mat_shape, Shape img_shape, int xPatchSize, int yPatchSize, int xPatches, int yPatches){
	// amount that x and y overhang by
	const int x_hang = ((img_shape.w() - 1) % xPatchSize) + 1;
	const int y_hang = ((img_shape.h() - 1) % yPatchSize) + 1;

	const int img_slice = img_shape.w() * img_shape.h();

	const int patch_size = xPatchSize * yPatchSize;

	// loop over rows of patches in input, besides last row
	for(int y = 0; y < yPatches - 1; y++){
		// loop over columns of patches in output, besides last column
		for(int x = 0; x < xPatches - 1; x++){
			// copy patch to output
			for(int d = 0; d < img_shape.d(); d++)
				vDSP_mmov(img + img_slice * d, mat + patch_size * d, xPatchSize, yPatchSize, img_shape.w(), xPatchSize);
			// move output to next row
			mat += mat_shape.w();
			// move input to patch column
			img += xPatchSize;
		}

		// process last column separately in case of 'overhang'
		// zero output
		memset(mat, 0, patch_size * img_shape.d() * sizeof(float));
		for(int d = 0; d < img_shape.d(); d++)
			vDSP_mmov(img + img_slice * d, mat + patch_size * d, x_hang, yPatchSize, img_shape.w(), xPatchSize);
		
		// move to next output row
		mat += mat_shape.w();

		// move input to next row
		img += img_shape.w() * (yPatchSize - 1) + x_hang;
	}

	// ===== process last row in input =====
	// loop over columns of patches in output, besides last column
	for(int x = 0; x < xPatches - 1; x++){
		memset(mat, 0, patch_size * img_shape.d() * sizeof(float));
		// copy patch to output
		for(int d = 0; d < img_shape.d(); d++)
			vDSP_mmov(img + img_slice * d, mat + patch_size * d, xPatchSize, y_hang, img_shape.w(), xPatchSize);
		// move output to next row
		mat += mat_shape.w();
		// move input to patch column
		img += xPatchSize;
	}

	// process last row and column separately in case of 'overhang'
	memset(mat, 0, patch_size * img_shape.d() * sizeof(float));
	for(int d = 0; d < img_shape.d(); d++)
		vDSP_mmov(img + img_slice * d, mat + patch_size * d, x_hang, y_hang, img_shape.w(), xPatchSize);
}

}