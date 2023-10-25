#ifndef IMG_FLATTEN_HEADER
#define IMG_FLATTEN_HEADER

#include "../layer.hpp"

namespace CPPML {

/* 
 * Flattens a 2d image into a form that that can be passed
 * into an attention layer (adds position embeddings).
 * The first layer added as input is the image to flatten
 * the remaining layers are used as embeds added after each
 * row of the flattened image.
 */
class ImageFlatten : public Layer {
public:
	// number of patches in the x and y directions
	int xPatches, yPatches;
	// size of patches in the x and y directions
	int xPatchSize, yPatchSize;
	// size of the sin embedding for x and y dimensions
	int xEmbSize, yEmbSize;

	int extra_embed_size;

	Layer* img_in;
	Shape image_shape;

	/// @param xPatchSize width of image embed patch size
	/// @param yPatchSize height of image embed patch size
	/// @param xEmbSize size of the embed of the x coordinate of image location (multiple of 2)
	/// @param yEmbSize size of the embed of the y coordinate of image location (multiple of 2)
	/// @param image_layer layer that is the source for the image to flatten
	/// @param embeds vararg, additional embeds to add to each row
	template<typename... Ts>
	ImageFlatten(int xPatchSize, int yPatchSize, int xEmbSize, int yEmbSize, Layer* image_layer, Ts... embeds) : Layer(embeds...){
		Init(xPatchSize, yPatchSize, xEmbSize, yEmbSize, image_layer);
	}

	virtual void populate(float* params, float* gradients);

	virtual std::string get_type_name(){return "Image_Flatten";}

private:
	// initialize layer
	void Init(int xPatchSize, int yPatchSize, int xEmbSize, int yEmbSize, Layer* l);

	virtual void compute(float* input, float* output, float* intermediate_buffer);

	virtual bool compile_();

	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);
};

/*
 * Takes a flattened image and transforms it back into
 * the given shape.
 */
class ImageDeFlatten : public Layer {
public:
	// number of patches in the x and y directions
	int xPatches, yPatches;
	// size of patches in the x and y directions
	int xPatchSize, yPatchSize;

	ImageFlatten* flatten_layer;

	/// @param xPatchSize width of image embed patch size
	/// @param yPatchSize height of image embed patch size
	/// @param imgw width of the output image
	/// @param imgh height of the output image
	/// @param imgd depth of the output image
	/// @param input *optional* singular input to this layer
	ImageDeFlatten(int xPatchSize, int yPatchSize, int imgw, int imgh, int imgd=1, Layer* input=nullptr);
	
	// gets necessary inputs from provided flatten layer, also
	// optionally takes in an input layer

	/// @param flatten_layer layer that flattened the input to this layer, necessary values will be sourced from here
	/// @param input *optional* singular input to this layer
	ImageDeFlatten(ImageFlatten* flatten_layer, Layer* input=nullptr);

	virtual void populate(float* params, float* gradients);

	virtual std::string get_type_name(){return "Image_DeFlatten";}

private:
	void init(int xPatchSize_, int yPatchSize_, int imgw, int imgh, int imgd, Layer* l);

	virtual void compute(float* input, float* output, float* intermediate_buffer);

	virtual bool compile_();
	
	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate);
};

}

#endif