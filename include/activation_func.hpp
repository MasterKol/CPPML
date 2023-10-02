#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <string>

#include "layer.hpp"

namespace CPPML {

struct ActivationFunc {
	void (*f)(const float* input, float* output, int num);
	void (*df)(const float* input, float* output, int num);
};

void linear_f(const float* input, float* output, int num);
void linear_df(const float* input, float* output, int num);

const ActivationFunc linear_org = {linear_f, linear_df};
const ActivationFunc* const LINEAR = &linear_org;

void elu_f(const float* input, float* output, int num);
void elu_df(const float* input, float* output, int num);

const ActivationFunc elu_org = {elu_f, elu_df};
const ActivationFunc* const ELU = &elu_org;

void relu_f(const float* input, float* output, int num);
void relu_df(const float* input, float* output, int num);

const ActivationFunc relu_org = {relu_f, relu_df};
const ActivationFunc* const RELU = &relu_org;

void sigmoid_f(const float* input, float* output, int num);
void sigmoid_df(const float* input, float* output, int num);

const ActivationFunc sigmoid_org = {sigmoid_f, sigmoid_df};
const ActivationFunc* const SIGMOID = &sigmoid_org;

void softmax_f(const float* input, float* output, int num);
void softmax_df(const float* input, float* output, int num);

const ActivationFunc softmax_org = {softmax_f, softmax_df};
const ActivationFunc* const SOFTMAX = &softmax_org;

}

#endif