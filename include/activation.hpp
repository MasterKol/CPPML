#ifndef ACTIVATION_H
#define ACTIVATION_H

//#include "layer.hpp"

namespace CPPML {

/*class ActivationLayer : public Layer {
	Activation act;
	ActivationLayer(const Activation* const activation);

	template<typename... Ts>
	ActivationLayer(const Activation* const activation_, Ts... input_layers) : Layer(input_layers...){
		init(nodes, activation_);
	}

	// gets pointer to parameter memory from
	// network and fills it with initial params
	virtual void populate(float* params, float* gradients);

	// returns the name of this layer type
	// wish there was a better way to do this
	// but there isn't as far as I know
	virtual std::string get_type_name(){return "Dense";}
private:
	// initialize layer
	void init(int nodes, const Activation* const activation);

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
};*/

struct Activation {
	void (*f)(const float* input, float* output, int num);
	void (*df)(const float* input, float* output, int num);
};

void linear_f(const float* input, float* output, int num);
void linear_df(const float* input, float* output, int num);

const Activation linear_org = {linear_f, linear_df};
const Activation* const LINEAR = &linear_org;

void elu_f(const float* input, float* output, int num);
void elu_df(const float* input, float* output, int num);

const Activation elu_org = {elu_f, elu_df};
const Activation* const ELU = &elu_org;

void relu_f(const float* input, float* output, int num);
void relu_df(const float* input, float* output, int num);

const Activation relu_org = {relu_f, relu_df};
const Activation* const RELU = &relu_org;

void sigmoid_f(const float* input, float* output, int num);
void sigmoid_df(const float* input, float* output, int num);

const Activation sigmoid_org = {sigmoid_f, sigmoid_df};
const Activation* const SIGMOID = &sigmoid_org;

void softmax_f(const float* input, float* output, int num);
void softmax_df(const float* input, float* output, int num);

const Activation softmax_org = {softmax_f, softmax_df};
const Activation* const SOFTMAX = &softmax_org;

}

#endif