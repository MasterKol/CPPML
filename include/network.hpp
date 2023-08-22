#ifndef NETWORK_H
#define NETWORK_H

namespace CPPML {
	class Network;
}

#include "optimizer.hpp"
#include "cost_func.hpp"
#include "optimizer.hpp"
#include "layer.hpp"
#include "data.hpp"
#include "Layers/input.hpp"
#include <string>
#include <vector>
#include <atomic>
#include <string>

namespace CPPML {

class Network {
public:
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

	int input_length, output_length;

	int num_params; // total parameters across all layers
	float* params;

	float* gradients;

	//float *lio, *inter, *change;

	std::atomic_int num_examples{0};

	Network(const Cost_func* const cost_func_, std::string name="model");

	// adds a layer to the network that is used for input
	// these layers should have no inputs themselves
	void add_input_layer(Input* input_layer);

	// Uses the layers that were added and finalizes them.
	// Does all of the setup needed to get the network working
	// including telling all layers to initialize themselves with
	// random parameters
	void compile(Optimizer* optimizer_);

	// Writes network output to out pointer, reads
	// data from subsequent args. If only one arg
	// is provided all data is read from there, otherwise
	// one pointer is required for each input layer and
	// should be provided in the same order as input layers
	// were added to this network.
	void eval(float* input, float* output, float* lio=NULL);

	// examples: array of examples with length = num * input  size
	// targets : array of targets  with length = num * output size
	// num: the number of training examples in the given arrays
	// calls fit_network(float*, float*) for each training example
	// in its own thread to speed up training
	void fit_network(float** examples, float** targets, int num);
	
	// calls fit_network with pointers to the start of
	// examples and targets as if they were float**s
	void fit_network(float* examples, float* targets, int num, float* loss=NULL);

	// Takes in a network input and a desired output and adds the
	// gradients for that example to the network. Thread safe
	void fit_network(float* example, float* target, float* lio=NULL, float* inter=NULL, float* change=NULL, float* loss=NULL);

	// applies gradients from previous training.
	// zeroes gradients and resets num_examples when done
	void apply_gradients();

	// prints a summary of the current network
	// only works after net is compiled
	void print_summary();

	// writes model weights to designated file
	void save(std::string file_name);

	// reads model weights from designated file
	void load(std::string file_name);
private:
	// This runs basically dfs topological sort on the nodes
	// in the network so that each one will only rely on
	// nodes that will have previously been processed
	void order_layers();

	// fits network in a separate thread
	void fit_network_thread(std::atomic_int* i, float** examples, float** targets);
};

}

#endif