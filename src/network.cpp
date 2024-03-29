#include "network.hpp"

#include <stdio.h>

#include <cstdlib>
#include <cassert>
#include <thread>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>

#include "LinearAlgebra.hpp"
#include "random.hpp"

#if defined(__has_include) && __has_include(<unistd.h>)
#include <unistd.h>
int myisatty(int fd){
	return isatty(fd);
}
#else
#include <io.h> // for windows
int myisatty(int fd){
	return _isatty(fd);
}
#endif

//#include </usr/local/opt/libomp/include/omp.h>

namespace CPPML {

Network::Network(const Cost_func* const cost_func_, float ema_decay_rate_, std::string name){
	cost_func = cost_func_;
	num_layers = 0;
	last_io_size = 0;
	num_params = 0;
	output_layer = nullptr;
	output_length = 0;
	input_length = 0;
	train_callback = nullptr;
	ema_decay_rate = ema_decay_rate_;
	ema_params = nullptr;
	params_ema = false;

	gradients = nullptr;
	params = nullptr;
	num_examples = 0;
	net_name = name;
}

void Network::add_input_layer(Input* input_layer){
	input_layers.push_back(input_layer);
}

void Network::set_params_to_ema(){
	if((!ema_params) || params_ema)
		return;
	vDSP_vswap(params, 1, ema_params, 1, num_params);
	params_ema = true;
}

void Network::set_params_to_norm(){
	if((!ema_params) || !params_ema)
		return;
	vDSP_vswap(params, 1, ema_params, 1, num_params);
	params_ema = false;
}

void Network::compile(Optimizer* optimizer_){
	// expands all layers in the network if necessary
	for(Layer* l : input_layers){
		l->expand();
	}

	// Find singular output layer of the network
	// this works because the network is directed and acyclic
	// and must have only one node of out-degree 0
	output_layer = input_layers[0];
	while(output_layer->outputs.size() > 0){
		output_layer = output_layer->outputs[0];
	}

	order_layers();

	// loop over all of the input layers and check
	// if they are actually proper input layers.
	// Also add up length of inputs
	input_length = 0;
	for(Layer* inLayer : input_layers){
		assert(inLayer->inputs.size() == 0);
		input_length += inLayer->output_shape.size();
	}

	// loop over all layers and compile them, then
	// add space to the buffer for their outputs.
	// Also add up the total number of parameters
	last_io_size = 0;
	intermediate_size = 0;
	for(Layer* layer : layers){
		layer->compile(last_io_size, intermediate_size);

		last_io_size += layer->output_shape.size();
		num_params += layer->num_params;
		intermediate_size += layer->intermediate_num;
	}

	// allocate parameter array for use by all layers
	if(!params) // make it conditional to allow for weight sharing between networks
		params = new float[num_params]();

	// allocate ema array for use by all layers
	if(ema_decay_rate != 0)
		ema_params = new float[num_params]();

	// allocate gradient array for use by all layers
	gradients = new float[num_params]();

	// loop over all layers and give them a pointer to their segment of
	// the parameter/gradient memory and tell them to initialize it
	float* layer_prms = params;
	float* layer_grds = gradients;
	for(Layer* layer : layers){
		layer->populate(layer_prms, layer_grds);

		layer_prms += layer->num_params;
		layer_grds += layer->num_params;
	}

	// copy params to ema_params if it exists
	if(ema_params)
		memcpy(ema_params, params, num_params * sizeof(float));

	// compile optimizer after all layers are compiled so
	// that it has information about how many params
	// are in the network
	optimizer = optimizer_;
	if(optimizer)
		optimizer->compile(this);

	// set output_shape
	output_length = output_layer->output_shape.size();
}

void Network::order_layers(){
	std::vector<Layer*> active;
	for(Layer* il : input_layers){
		// make sure that input layers have no inputs themselves
		assert(il->inputs.size() == 0);

		// for each node this input node outputs to, add it to the
		// active list and add one to its satisfied number,
		// temp stored in the output index
		layers.push_back(il);
		for(Layer* ol : il->outputs){
			if(ol->output_index == 0){
				active.push_back(ol);
			}
			ol->output_index++;
		}
	}

	// only break when there are no nodes left to remove
	while(active.size() > 0){
		int i = active.size() - 1;
		for(; i >= 0; i--){
			Layer* act = active[i];
			// if node is not valid, go next
			if(act->output_index != act->inputs.size()){
				continue;
			}
			// this layer is ready to be added

			layers.push_back(act);

			// add next nodes to active list
			for(Layer* ol : act->outputs){
				if(ol->output_index == 0){
					active.push_back(ol);
				}
				ol->output_index++;
			}

			// remove processed node
			active.erase(active.begin() + i);
			break; // break because array has changed
		}

		// if i == -1 then then no nodes were removed
		// this means that there is a cycle in the network
		if(i == -1){
			for(int j = 0; j < active.size(); j++){
				printf("%d, %s\n", j, active[j]->get_type_name().c_str());
			}
		}
		assert(i != -1);
	}
}

void Network::eval(float* input, float* output, float* lio_){
	// create memory for storing network io
	float* lio = lio_;
	if(lio_ == nullptr){
		lio = new float[last_io_size];
	}

	// copy inputs to lio
	memcpy(lio, input, input_length * sizeof(float));

	// process input through each layer
	for(Layer* l : layers){
		l->process(lio);
	}

	// copy output from lio to
	memcpy(output, lio + output_layer->output_index, output_length * sizeof(float));

	if(lio_ == nullptr){
		delete[] lio;
	}
}

void Network::fit_network(float* examples, float* targets, int num, float* loss){
	float temp_loss = 0;
	#pragma omp parallel reduction(+ : temp_loss)
	{
		std::unique_ptr<float[]> lio 	(  new float[last_io_size]		);
		std::unique_ptr<float[]> inter	(  new float[intermediate_size] );
		std::unique_ptr<float[]> change	(  new float[last_io_size]		);
		if(loss == nullptr){
			#pragma omp for
			for(int i = 0; i < num; i++){
				fit_network(examples + i * input_length, targets + i * output_length, lio.get(), inter.get(), change.get());
			}
		}else{
			#pragma omp for
			for(int i = 0; i < num; i++){
				float t;
				fit_network(examples + i * input_length, targets + i * output_length, lio.get(), inter.get(), change.get(), &t);
				temp_loss += t;
			}
		}
	}
	if(loss)
		*loss = temp_loss;
}

void Network::fit_network(float* example, float* target, float* lio_, float* inter_, float* change_, float* loss){
	// create memory for storing network io
	float* lio = lio_;
	if(!lio_){
		lio = new float[last_io_size];
	}
	// create memory for intermediate vals
	float* inter = inter_;
	if(!inter_){
		inter = new float[intermediate_size];
	}
	memset(inter, 0, intermediate_size * sizeof(float));
	
	// copy example to lio
	memcpy(lio, example, input_length * sizeof(float));

	// process input through each layer and get
	// intermediate values
	for(Layer* l : layers){
		l->process(lio, inter, true);
	}

	// create mem to store change for back prop
	float* change = change_;
	if(!change_){
		change = new float[last_io_size];
	}
	memset(change, 0, last_io_size * sizeof(float)); // zero change

	// get derivative of cost function and write it
	// to the last part of change to start backprop
	const int oi = output_layer->output_index;
	cost_func->get_cost_derv(lio + oi, target, change + oi, output_length);

	if(loss != nullptr){
		*loss = cost_func->get_cost(lio + oi, target, output_length);
	}
	
	// iterate over layers backwards and backpropagate through them
	for(auto l = layers.rbegin(); l != layers.rend(); l++){
		(*l)->backpropagate(change, lio, inter);
	}

	if(train_callback)
		train_callback(this, example, target, loss, lio, inter, change);

	// free memory if it was created locally
	if (!lio_)
		delete[] lio;
	if (!inter_)
		delete[] inter;
	if (!change_)
		delete[] change;

	num_examples++;
}

void Network::apply_gradients(){
	// if no examples have been trained nothing will happen
	// so save some processing
	if(num_examples == 0){
		return;
	}

	// divide gradients by the number of examples
	float invNumExamps = 1.0f / (float)num_examples;
	//printf("NUM EXAMPS: %d, %f\n", (int)num_examples, invNumExamps);
	vDSP_vsmul(gradients, 1, &invNumExamps, gradients, 1, num_params);

	// tell optimizer to update parameters
	optimizer->update_params();

	// reset num examples and zero gradients
	num_examples = 0;
	memset(gradients, 0, num_params * sizeof(float));

	if(ema_params){
		float n_ema_m1 = 1 - ema_decay_rate;
		vDSP_vsmsma(ema_params, 1, &ema_decay_rate, params, 1, &n_ema_m1, ema_params, 1, num_params);
	}
}

float Network::get_loss(float* input, float* target){
	std::unique_ptr<float[]> output(new float[output_length]);
	eval(input, output.get());
	return cost_func->get_cost(output.get(), target, output_length);
}

float Network::get_loss(float* inputs, float* targets, int num){
	std::unique_ptr<float[]> output(new float[output_length]);
	float out = 0;
	for(int i = 0; i < num; i++){
		eval(inputs, output.get());
		out += cost_func->get_cost(output.get(), targets, output_length);

		inputs += input_length;
		targets += output_length;
	}
	return out / num;
}

Network::Err Network::save(std::string file_name, bool save_ema){
	// open file as output, binary, and delete original content
	std::ofstream file (file_name, std::ios::out|std::ios::binary|std::ios::trunc);
	if (!file.is_open())
		return file_not_found;

	uint t = htonl(num_params);
	// write number of parameters
	file.write((const char*)(&t), 4);

	// tell compiler to read params as uint instead of float
	uint* u_params = (uint*)params;

	if(ema_params && ((!params_ema) ^ save_ema)) // if ema is enabled, save ema rather than base parameters
		u_params = (uint*)ema_params;

	// loop over params
	for(int i = 0; i < num_params; i++){
		// convert from local endian to big endian
		t = htonl(u_params[i]);
		file.write((const char*)(&t), 4);
	}

	file.close();
	return success;
}

Network::Err Network::load(std::string file_name, bool load_only_ema){
	if(load_only_ema && !ema_params)
		return Err::wrong_param_num;
	
	// open file as output, binary, and delete original content
	std::ifstream file (file_name, std::ios::in|std::ios::binary);
	if (!file.is_open())
		return file_not_found;

	uint t;

	// read number of parameters
	file.read((char*)(&t), 4);
	t = ntohl(t);

	if(t != num_params){
		return wrong_param_num;
	}

	// tell compiler to read params as uint instead of float
	uint* u_params = (uint*)params;
	// load into ema_params if load ema is true
	if(load_only_ema)
		u_params = (uint*)ema_params;

	// load params from file
	file.read((char*)(u_params), sizeof(float) * t);

	// convert from big endian to local endian
	for(int i = 0; i < num_params; i++){
		u_params[i] = ntohl(u_params[i]);
	}

	file.close();

	if(!load_only_ema && ema_params)
		memcpy(ema_params, params, num_params * sizeof(float));

	return success;
}

std::string get_formatted_name(Layer* l){
	std::string out = l->name;
	if(out.length() > 0){
		out += " ";
	}
	out += "(" + l->get_type_name() + ")";
	return out;
}

inline int get_layer_ind(Layer* l, std::vector<Layer*> layers){
	for(int i = 0; i < layers.size(); i++){
		if (l == layers[i])
			return i;
	}
	return -1;
}

std::string get_layer_input_string(Layer* l, std::vector<Layer*> layers){
	std::string out = "[";
	for(Layer* il : l->inputs){
		if(il->name.length() == 0){
			out += "(" + il->get_type_name() + ")";
		}else{
			out += "\"" + il->name + "\"";
		}
		out += "[" + std::to_string(get_layer_ind(il, layers)) + "], ";
	}
	out = out.substr(0, out.length() - 2) + "]";
	return out;
}

void set_column_sizes(int terminal_width, int num_columns, int* column_sizes){
	int finalized = 0;
	int assigned_length = num_columns + 2; // initial assigned space is for inter-column spaces
	while(finalized < num_columns){
		int per_row = (terminal_width - assigned_length) / (num_columns - finalized);
		bool assigned = false;
		for(int i = 0; i < num_columns; i++){
			if(column_sizes[i] >= 0) continue;

			if(-column_sizes[i] <= per_row){
				column_sizes[i] = -column_sizes[i];
				assigned_length += column_sizes[i];
				finalized++;
				assigned = true;
			}
		}
		if (!assigned) break;
	}

	if(finalized == num_columns){ // all columns were assigned to, allocate extra space
		int extra_space = (terminal_width - assigned_length);
		int per_row = extra_space / num_columns;
		int xtra_cols = extra_space % num_columns;
		// assign an extra 1 char to the first few rows
		for(int i = 0; i < num_columns; i++){
			column_sizes[i] += per_row + (i < xtra_cols);
		}
	}else{ // not all columns were assigned distribute remaining space evenly
		int extra_space = (terminal_width - assigned_length);
		int per_row = extra_space / (num_columns - finalized);
		int xtra_cols = extra_space % (num_columns - finalized);

		for(int i = 0; i < num_columns; i++){
			if(column_sizes[i] >= 0) continue;
			column_sizes[i] = per_row + (xtra_cols > 0);
			xtra_cols--;
		}
	}
}

void print_centered(std::string s, int width){
	int extra = width - s.length();
	int l_size = extra / 2;
	std::cout << std::string(l_size, ' ') << s << std::string(extra - l_size, ' ');
}

/* copied from this stack overflow:
 * https://stackoverflow.com/questions/23369503/get-size-of-terminal-window-rows-columns
 */
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__) || __APPLE__
#include <sys/ioctl.h>
#endif // Windows/Linux/Mac

/*
 * returns the width of the current terminal window
*/
int get_terminal_width(){
	#if defined(_WIN32)
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
		return (int)(csbi.srWindow.Right-csbi.srWindow.Left+1);
	#elif defined(__linux__) || __APPLE__
		struct winsize w;
		ioctl(fileno(stdout), TIOCGWINSZ, &w);
		return (int)(w.ws_col);
	#endif // Windows/Linux/Mac
}
/*end copied section*/

void Network::print_summary(){
	// The output format is lifted from tensorflow for
	// similarities sake
	int terminal_width = 200;
	if(myisatty( fileno(stdout) )) // check if stdout is a terminal or not
		terminal_width = get_terminal_width();

	std::cout << "Model: " << net_name << std::endl;
	std::cout << std::string(terminal_width, '-') << std::endl;

	const int num_columns = 6;

	std::string column_headers[] = {"Num", "Layer (type)", "Input Shape", "Output Shape", "Param #", "Inputs"};
	// layer_names, output_shapes, param_nums, inputs
	std::vector<std::string> columns[num_columns];

	int c = 0;
	for(Layer* l : layers){
		columns[0].push_back(std::to_string(c++));
		columns[1].push_back(get_formatted_name(l));
		columns[2].push_back(l->input_shape.to_string());
		columns[3].push_back(l->output_shape.to_string());
		columns[4].push_back(std::to_string(l->num_params));
		columns[5].push_back(get_layer_input_string(l, layers));
	}

	int column_sizes[num_columns];
	for(int i = 0; i < num_columns; i++){
		column_sizes[i] = -column_headers[i].length();
		for(int j = 0; j < layers.size(); j++){
			column_sizes[i] = std::min(column_sizes[i], -(int)columns[i][j].length());
		}
	}

	set_column_sizes(terminal_width, num_columns, column_sizes);

	// print headers
	std::cout << ' ';
	for(int i = 0; i < num_columns; i++){
		std::cout << std::left << std::setw(column_sizes[i]) << std::setfill(' ') << column_headers[i] << " ";
	}
	std::cout << std::endl;
	std::cout << std::string(terminal_width, '=') << std::endl;

	for(int i = 0; i < layers.size(); i++){
		bool finished;
		int sc = 0;
		do{
			std::cout << ' ';
			finished = true;
			for(int j = 0; j < num_columns; j++){
				if ((int)columns[j][i].length() - sc * column_sizes[j] > 0){
					//std::cout << j << ", " << sc << ", " << columns[j][i].length() << ", " << sc * column_sizes[j] << std::endl;
					finished = false;
					std::cout << std::left << std::setw(column_sizes[j]) << std::setfill(' ') << columns[j][i].substr(sc * column_sizes[j], std::max((sc+1) * column_sizes[j], (int)columns[j][i].length() - 1)) << " ";
				}else{
					std::cout << std::string(column_sizes[j] + 1, ' ');
				}
			}
			std::cout << std::endl;
			sc++;
		}while(!finished);
	}
	std::cout << std::string(terminal_width, '=') << std::endl;

	std::cout << "Total Params: " << num_params << std::endl;

	std::cout << std::string(terminal_width, '-') << std::endl;
}

}