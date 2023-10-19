#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <string>

#include "layer.hpp"

namespace CPPML {

/*
 * Defines an arbitrary function that can be used as an activation function
 * or simply as a transformation such as mx+b
 */
struct ActivationFunc {
	/// @brief evaluate activation function
	/// @param input input value to the function
	/// @param output output of function
	/// @param length number of elements in the array
	void (*f)(const float* input, float* output, int length);

	/// @brief derivative of the function
	/// @param input input to the function
	/// @param input_gradients output, derivative of the function w.r.t. the inputs and downstream derivatives
	/// @param output output of the function at the given input, may be changed
	/// @param output_gradients gradient of outputs, may equal input_gradients
	/// @param length length of the vector
	void (*df)(const float* input, float* input_gradients, float* output, float* output_gradients, int length);
};

void linear_f(const float* input, float* output, int length);
void linear_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length);

const ActivationFunc linear_org = {linear_f, linear_df};
const ActivationFunc* const LINEAR = &linear_org;

void elu_f(const float* input, float* output, int length);
void elu_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length);

const ActivationFunc elu_org = {elu_f, elu_df};
const ActivationFunc* const ELU = &elu_org;

void relu_f(const float* input, float* output, int length);
void relu_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length);

const ActivationFunc relu_org = {relu_f, relu_df};
const ActivationFunc* const RELU = &relu_org;

void sigmoid_f(const float* input, float* output, int length);
void sigmoid_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length);

const ActivationFunc sigmoid_org = {sigmoid_f, sigmoid_df};
const ActivationFunc* const SIGMOID = &sigmoid_org;

void softmax_f(const float* input, float* output, int length);
void softmax_df(const float* input, float* input_gradients, float* output, float* output_gradients, int length);

const ActivationFunc softmax_org = {softmax_f, softmax_df};
const ActivationFunc* const SOFTMAX = &softmax_org;

}

#endif