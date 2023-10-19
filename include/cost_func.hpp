#ifndef COST_HEADER
#define COST_HEADER

namespace CPPML {

/*
 * Defines a cost function used for determining the error
 * between predicted and ground truth values.
 */
struct Cost_func {
	/// @brief gets the cost between x and y
	/// @param x array of predictions
	/// @param y array of ground truth values
	/// @param length length of x and y
	float (*get_cost)(float* x, float* y, int length);
	
	/// @brief gets the derivative of the cost between x and y
	/// @param x array of predictions
	/// @param y array of ground truth values
	/// @param out place to write derivatives
	/// @param length length of x, y, and out
	void (*get_cost_derv)(float* x, float* y, float* out, int length); // places the cost derivative of x and y in out
};

/**************** Mean Squared Error ****************/
float mse_get_cost(float* x, float* y, int length);
void mse_get_cost_derv(float* x, float* y, float* out, int length);

const Cost_func MSE_org = {mse_get_cost, mse_get_cost_derv};
const Cost_func* const MSE = &MSE_org;

/**************** Mean Absolute Error ****************/
float mae_get_cost(float* x, float* y, int length);
void mae_get_cost_derv(float* x, float* y, float* out, int length);

const Cost_func MAE_org = {mae_get_cost, mae_get_cost_derv};
const Cost_func* const MAE = &MAE_org;

/**************** Huber ****************/
float huber_get_cost(float* x, float* y, int length);
void huber_get_cost_derv(float* x, float* y, float* out, int length);

const Cost_func HUBER_org = {huber_get_cost, huber_get_cost_derv};
const Cost_func* const HUBER = &HUBER_org;

/**************** CrossEntropy ****************/
float cross_entropy_get_cost(float* x, float* y, int length);
void cross_entropy_get_cost_derv(float* x, float* y, float* out, int length);

const Cost_func CROSS_ENTROPY_org = {cross_entropy_get_cost, cross_entropy_get_cost_derv};
const Cost_func* const CROSS_ENTROPY = &CROSS_ENTROPY_org;

}

#endif