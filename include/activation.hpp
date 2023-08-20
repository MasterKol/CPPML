#ifndef ACTIVATION_H
#define ACTIVATION_H

namespace CPPML {

struct Activation {
	void (*f)(float* input, float* output, int num);
	void (*df)(float* input, float* output, int num);
};

void linear_f(float* input, float* output, int num);
void linear_df(float* input, float* output, int num);

const Activation linear_org = {linear_f, linear_df};
const Activation* const LINEAR = &linear_org;

void elu_f(float* input, float* output, int num);
void elu_df(float* input, float* output, int num);

const Activation elu_org = {elu_f, elu_df};
const Activation* const ELU = &elu_org;

void relu_f(float* input, float* output, int num);
void relu_df(float* input, float* output, int num);

const Activation relu_org = {relu_f, relu_df};
const Activation* const RELU = &relu_org;

void sigmoid_f(float* input, float* output, int num);
void sigmoid_df(float* input, float* output, int num);

const Activation sigmoid_org = {sigmoid_f, sigmoid_df};
const Activation* const SIGMOID = &sigmoid_org;

}

#endif