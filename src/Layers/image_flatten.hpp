#ifndef IMG_FLATTEN_HEADER
#define IMG_FLATTEN_HEADER

#include "../layer.hpp"

/* 
 * Flattens a 2d image into a form that that can be passed
 * into an attention layer (adds position embeddings)
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

	/*template<typename... Ts>
	ImageFlatten(int xPatchSize, int yPatchSize, int xEmbSize, int yEmbSize, int iw=-1, int ih=-1, Layer* l){
		Init(xPatchSize, yPatchSize, xEmbSize, yEmbSize, iw, ih, l);
	}*/

	template<typename... Ts>
	ImageFlatten(int xPatchSize, int yPatchSize, int xEmbSize, int yEmbSize, Layer* l, Ts... embeds) : Layer(embeds...){
		Init(xPatchSize, yPatchSize, xEmbSize, yEmbSize, l);
	}

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "Image_Flatten";}

private:
	// initialize layer
	void Init(int xPatchSize, int yPatchSize, int xEmbSize, int yEmbSize, Layer* l);

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

class ImageDeFlatten : public Layer {
public:
	// number of patches in the x and y directions
	int xPatches, yPatches;
	// size of patches in the x and y directions
	int xPatchSize, yPatchSize;

	ImageDeFlatten(int xPatchSize, int yPatchSize, int imgw, int imgh, Layer* l=NULL);
	
	// gets necessary inputs from provided flatten layer, also
	// optionally takes in an input layer
	ImageDeFlatten(ImageFlatten* flatten_layer, Layer* l=NULL);

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "Image_DeFlatten";}

private:
	void init(int xPatchSize_, int yPatchSize_, int imgw, int imgh, Layer* l);

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

#endif