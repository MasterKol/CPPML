#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <mutex>
#include <cassert>
#include <string>

#include "shape.hpp"

namespace CPPML {

/*
 * Layer interface, all layers in a network extend this.
 * Layer* can be used for all types of layers.
 */
class Layer {
public:
	Shape input_shape, output_shape;
	std::vector<Layer*> inputs, outputs;

	// number of variable parameters this layer has
	int num_params;

	// size of intermediate values that need to be
	// stored during training
	int intermediate_num;

	// index in intermediate buffer
	int intermediate_index;

	// index of start of outputs in output buffer
	int output_index;

	// name of this layer
	std::string name;

protected:
	// mutex to protect gradients while they are being modified
	std::mutex gradient_mutex;

	// is the layer expanded or not?
	bool expanded;

	// true if a layer wants to use batch processing
	bool is_batch_processing;

public:
	/// @brief 
	/// @param input_layers vararg adds given Layer*'s as inputs to this layer
	template<typename... Ts>
	Layer(Ts... input_layers){
		num_params = 0;
		output_index = 0;
		input_shape = Shape(-1);
		intermediate_num = 0;
		intermediate_index = 0;
		name = "";
		expanded = false;
		is_batch_processing = false;

		(add_input(input_layers), ...);
	}

	/// @brief Set network name
	/// @param name new name of the layer
	/// @return pointer to self
	Layer* set_name(std::string name);

	/// @brief returns the name of this layer type
	virtual std::string get_type_name() = 0;

	/// @brief adds given layer as new input
	/// @param layer layer to add
	void add_input(Layer* layer);

	// returns the layer that represents the output
	// of this layer, usually 'this' but in some cases
	// may be different (defaults to 'this')
	Layer* get_output();

	// call for use by network, sets up input and output
	// for this layer to read and write from. If there are
	// multiple inputs it collects them together into
	// temporary memory

	/// @brief computes output of layer, pulls from input layers' input
	///		   locations in io_buffer and writes output to output_index
	/// @param io_buffer contains outputs of all layers in the network, in order
	/// @param intermediate_buffer contains intermediate values used only for training, null during inference
	void process(float* io_buffer, float* intermediate_buffer=nullptr);

	// can be over written to implement batch processing unique to a layer, only for use at training time

	void process_train(float* io_buffers, int buffer_len, int num, float* intermediate_buffers, int inter_buffer_len);

	/// @brief Compiles layer, does basic setup before calling layer specific compile_
	/// @param buffer_index index in io_buffer that outputs should be written to
	/// @param inter_index index in intermediate_buffer that intermediates should be written
	void compile(int buffer_index, int inter_index);

	// sets up inputs for get_change and add_gradients, collects
	// input and output changes, inputs and outputs, and sets 
	// offset for intermediate buffer finally writes output changes
	// to their proper place in the change buffer

	/// @brief propagates gradients backwards through the layer, at the
	///		   same time calculates gradients of this layers parameters
	/// @param change_buffer buffer where all network gradients are stored
	/// @param io_buffer buffer where all layer input and outputs are stored
	/// @param intermediate_buffer buffer where intermediate values needed by layer is stored
	void backpropagate(float* change_buffer, float* io_buffer,
					   float* intermediate_buffer);

	// gets pointer to parameter memory from network and
	// fills it with initial params. Also stores pointer to
	// gradients for the layer's parameters

	/// @brief Assigns layer is parameter and gradient memory and,
	///		   tells layer to initialize this memory
	/// @param params memory where this layers parameters are to be stored
	/// @param gradients memory where this layers parameter gradients are to be stored
	virtual void populate(float* params, float* gradients) = 0;

	/// @brief Calls expand_ for this layer and all children.
	void expand();
private:
	/// @brief Only ever called once
	/// @return true if expansion occurred, false otherwise
	virtual bool expand_();

	// collects inputs from the io buffer and writes them into
	// the provided array input should be at least
	// input_shape.size * sizeof(float) bytes long

	/// @brief collects inputs from the io buffer and writes them into the provided array
	/// @param io_buffer buffer storing all network layer io
	/// @param input array where layer inputs are written (size=input_shape.size())
	void collect_inputs(float* io_buffer, float* input);

	/// @brief performs this layer's computation reading from the input and writing to the output
	/// @param input input into the layer, contiguous
	/// @param output location to write layer output to
	/// @param intermediate_buffer location to write intermediate values (may be nullptr)
	virtual void compute(float* input, float* output, 
						 float* intermediate_buffer) = 0;

	virtual void batch_compute(float* inputs, float* outputs, float* intermediate_buffers);

	// sets up a layer given its inputs are already
	// compiled, only need to set i/o size and intermediate_num
	// returns true if layer is an input layer, false otherwise

	/// @brief sets up a layer given its inputs are already compiled,
	///		   only need to set i/o size and intermediate_num
	/// @return true if layer is an input layer, false otherwise
	virtual bool compile_() = 0;

	// takes in previous layer's change and calculates the change
	// of its inputs, WRITE TO inpt_change. inpt_change will always
	// be a copy so just write to it, adding is done externally
	// output, out_change, and intermediate can be changed because 
	// they will not be used downstream. Also calculates gradients
	// for this layer's params and adds them to its gradient buffer

	/// @brief takes in previous layer's change and calculates the change of its inputs
	///		   and the gradients of the layers parameters
	/// @param out_change derivative of output values (mutable)
	/// @param inpt_change derivative of input values, WRITE to this array
	/// @param input input to this layer
	/// @param output previous output of this layer (mutable)
	/// @param intermediate intermediate values saved during compute (mutable)
	virtual void get_change_grads(float* out_change, float* inpt_change,
				  float* input, float* output, float* intermediate) = 0;
};

}

#endif