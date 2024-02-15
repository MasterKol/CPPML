#include "layer.hpp"

#include <iostream>

#include "shape.hpp"
#include "LinearAlgebra.hpp"

namespace CPPML {

void Layer::add_input(Layer* layer){
	if(!layer)
		return;
	//inputs.push_back(layer->get_output());
	inputs.push_back(layer);
	
	// add this layer to inputs of the given layer
	layer->outputs.push_back(this);
}

Layer* Layer::get_output(){
	return this;
}

Layer* Layer::set_name(std::string name_){
	name = name_;
	return this;
}

void Layer::compile(int buffer_index, int inter_index){
	output_index = buffer_index;
	intermediate_index = inter_index;
	
	bool is_input = compile_();

	if(!is_input && inputs.size() == 0){
		// Layer has no inputs and is not an Input Layer
		std::cerr << "Layer has no inputs and is not an input Layer";
		exit(-1);
	}
}

void Layer::expand(){
	if(expanded)
		return;
	
	// try and expand
	bool didexpand = expand_();
	expanded = true;

	// recurse to children and expand them
	for(Layer* l : outputs){
		l->expand();
	}

	// expanding outputs is all that is needed if this
	// layer did not expand
	if(!didexpand)
		return;

	// call expand on inputs and expand them if
	// they are new
	for(Layer* l : inputs){
		l->expand();
	}
}

bool Layer::expand_(){
	return false;
}

void Layer::collect_inputs(float* io_buffer, float* input){
	for(Layer* l : inputs){ // copy data from each layer
		// FIXME, add option for choosing only part of input
		memcpy(input, io_buffer + l->output_index, l->output_shape.size() * sizeof(float));
		input += l->output_shape.size();
	}
}

void Layer::process(float* io_buffer, float* intermediate_buffer, bool training){
	float* output = io_buffer + output_index;
	float* intermediate = nullptr;

	if(intermediate_buffer != nullptr){
		intermediate = intermediate_buffer + intermediate_index;
	}

	switch(inputs.size()){
		case 0:
			// input layers have 0 inputs, the default one
			// does no computation but if someone extends
			// it they might want to do something...
			compute(nullptr, output, intermediate, training);
			return;
		case 1:
			// only one input so just pull directly from the io buffer
			float* input = io_buffer + inputs[0]->output_index;
			compute(input, output, intermediate, training);
			return;
	}

	// multiple inputs to copy to a temp buffer
	std::unique_ptr<float[]> input ( new float[input_shape.size()] );
	
	collect_inputs(io_buffer, input.get());

	compute(input.get(), output, intermediate, training);
}

void Layer::backpropagate(float* change_buffer, float* io_buffer, float* intermediate_buffer){
	// nowhere to push back to, simply return
	if(input_shape.size() <= 0){
		return;
	}
	float* out_change = change_buffer + output_index;
	float* input = nullptr;
	float* output = io_buffer + output_index;
	float* intermediate = intermediate_buffer + intermediate_index;

	switch(inputs.size()){
		case 1:
			// only one input so just pull directly from the io buffer
			input = io_buffer + inputs[0]->output_index;
		case 0:
			// input layers have 0 inputs, the default one
			// does no computation but if someone extends
			// it they might want to do something...
			break;
		default:
			// multiple inputs, collect them into input array
			input = new float[input_shape.size()];
			collect_inputs(io_buffer, input);
	}

	float* inpt_change = new float[input_shape.size()];
	memset(inpt_change, 0, input_shape.size() * sizeof(float)); // zero inpt_change

	// run layer specific get_change and add_gradients
	get_change_grads(out_change, inpt_change, input, output, intermediate);

	// put input change into correct places
	int offset = 0;
	for(Layer* l : inputs){
		float* write_pos = change_buffer + l->output_index; // pos to write to
		// add this layer's changes to the changes already present
		vDSP_vadd(inpt_change + offset, 1, write_pos, 1, write_pos, 1, l->output_shape.size());
		offset += l->output_shape.size();
	}

	delete[] inpt_change;

	// if there are multiple inputs the input array needs to be freed
	if(inputs.size() > 1){
		delete[] input;
	}
}

}