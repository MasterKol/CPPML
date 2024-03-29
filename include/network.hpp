#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include <vector>
#include <atomic>
#include <string>

#include "optimizer.hpp"
#include "cost_func.hpp"
#include "optimizer.hpp"
#include "layer.hpp"
#include "shape.hpp"
#include "Layers/input.hpp"

namespace CPPML {

class Input;
class Optimizer;

/************************************************************
 * Network build process:
 * 1.) Network object is created
 * 2.) Layers are added
 * 3.) User calls compile(optimizer*)
 * 	  	(Maybe include and expand() command to allow layers to change their input / outputs)
 *    a.) Find singular output layer
 *    b.) Layers are ordered according to DAG
 *    c.) Find and check all input layers
 *    d.) Each layer is compiled and its stats are recorded
 *    e.) Allocate memory and assign it to layers
 *    f.) compile optimizer
 ************************************************************/

/*
 * Stores a network and all of its contents, manages
 * gradients and optimizer during training.
 */
class Network {
public:
	enum Err {
		success = 1,
		file_not_found = -1,
		wrong_param_num = -2,
	};

	const Cost_func* cost_func;
	Optimizer* optimizer;

	std::vector<Input*> input_layers;
	std::vector<Layer*> layers;
	Layer* output_layer;
	std::string net_name;

	int num_layers;
	// sum of the length of the output of all layers
	int last_io_size;
	// size of the vector that stores layers' intermediate
	// values, only needed for training
	int intermediate_size;

	// total length of network input
	int input_length;
	// length of network output
	int output_length;

	float ema_decay_rate;

	// total number of parameters in the network
	int num_params;
	// all network parameters
	float* params;

	// exponential moving average of parameters
	float* ema_params;
	// are the values in params ema?
	bool params_ema;

	// gradient of network parameters
	float* gradients;

	// number of examples that the net has been trained on
	// since the last call to apply_gradients()
	std::atomic_int num_examples;

	/// @brief Callback called each time fit_network processes an example, may be called in parallel
	/// @param net this network
	/// @param example example used for this training iteration
	/// @param target target for this training iteration
	/// @param loss points to computed loss, nullptr if loss is not calculated
	/// @param last_io input and output values of all layers
	/// @param intermediate intermediate values used by all layers
	/// @param change local gradients of all layers, including inputs
	void (*train_callback)(Network* net, float* example, float* target, float* loss, float* last_io, float* intermediate, float* change);

	/// @brief Create new network
	/// @param cost_func Cost function used for network evaluation
	/// @param ema_decay_rate decay rate for exponential moving average, 0 is disabled (default)
	/// @param name Name of the model
	Network(const Cost_func* const cost_func, float ema_decay_rate=0.0f, std::string name="model");

	// adds a layer to the network that is used for input
	// these layers should have no inputs themselves
	
	/// @brief adds a layer to the network that is used for input
	/// @param input_layer Layer to add as input
	void add_input_layer(Input* input_layer);

	// Uses the layers that were added and finalizes them.
	// Does all of the setup needed to get the network working
	// including telling all layers to initialize themselves with
	// random parameters

	/// @brief Sets up the network, layers, and optimizer.
	///		   To be called only after all layers have been added.
	/// @param optimizer optimizer to use during network training
	void compile(Optimizer* optimizer);

	/// @brief Evaluates the network on the given input. For multi-input networks the input
	///		   should be concatenated in the order that the input layers were added.
	/// @param input Input to the network
	/// @param output Place to write network output
	/// @param lio *optional* memory where network intermediates are stored (size=last_io_size)
	void eval(float* input, float* output, float* lio=nullptr);

	/// @brief gets the loss between the predicted and target values for a given input
	/// @param input input to the model
	/// @param target target output for the model
	/// @return loss
	float get_loss(float* input, float* target);

	/// @brief gets the loss between array of predicted and target values for given inputs
	/// @param input inputs to the model
	/// @param target target outputs for the model
	/// @return average loss across inputs
	float get_loss(float* inputs, float* targets, int num);

	// moves ema parameters to params
	// if no ema, then nothing is done
	void set_params_to_ema();

	// moves params from ema back to params nothing is
	// done if params are in the proper place already.
	void set_params_to_norm();

	// examples: array of examples with length = num * input  size
	// targets : array of targets  with length = num * output size
	// num: the number of training examples in the given arrays
	// calls fit_network(float*, float*) for each training example
	// in its own thread to speed up training

	/// @brief Fits the network on the given values, runs in parallel
	/// @param examples pointers to input examples
	/// @param targets pointers to target for given examples
	/// @param num number of examples given
	void fit_network(float** examples, float** targets, int num);

	/// @brief  Fits the network on the given values, runs in parallel
	/// @param examples pointer to array of input examples
	/// @param targets pointer to array of targets for given examples
	/// @param num number of examples given
	/// @param loss optionally compute the sum training loss of the network on the provided examples
	void fit_network(float* examples, float* targets, int num, float* loss=nullptr);

	/// @brief Fit the network on a single training example, thread safe
	/// @param example pointer to example to train on
	/// @param target pointer to target value for given example
	/// @param lio *optional* memory where network intermediates are stored (size=last_io_size)
	/// @param inter *optional* memory where network intermediates are stored (size=intermediate_size)
	/// @param change *optional* memory where network layer gradients are stored (size=last_io_size)
	/// @param loss *optional* loss for this training example
	void fit_network(float* example, float* target, float* lio=nullptr, float* inter=nullptr, float* change=nullptr, float* loss=nullptr);

	/// @brief Applies gradients from previous training. Zeroes gradients and resets num_examples when done.
	void apply_gradients();

	/// @brief Prints a summary of the current network, only works after net is compiled.
	void print_summary();

	/// @brief Saves model weights to designated file
	/// @param file_name path to file to write to
	/// @param save_ema saves ema parameters rather than normal parameters if possible
	/// @return Returns error code if failure
	Err save(std::string file_name, bool save_ema=true);

	/// @brief Loads model weights from designated file
	/// @param file_name path to file to read from
	/// @param load_only_ema if false, loads into normal and ema params, if true loads only ema_params
	/// @return Returns error code if failure
	Err load(std::string file_name, bool load_only_ema=false);
private:
	// This runs basically dfs topological sort on the nodes
	// in the network so that each one will only rely on
	// nodes that will have previously been processed
	void order_layers();
};

}

#endif