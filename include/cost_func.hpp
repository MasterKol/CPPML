#ifndef COST_HEADER
#define COST_HEADER

typedef struct t_cost {
	float (*get_cost)(float* x, float* y, int length); // returns the cost between x and y
	void (*get_cost_derv)(float* x, float* y, float* out, int length); // places the cost derivative of x and y in out
} Cost_func;

/**************** Mean Squared Error ****************/
float mse_get_cost(float* x, float* y, int length);
void mse_get_cost_derv(float* x, float* y, float* out, int length);

const Cost_func MSE_org = (Cost_func){.get_cost=mse_get_cost, .get_cost_derv=mse_get_cost_derv};
const Cost_func* const MSE = &MSE_org;

/**************** Mean Absolute Error ****************/
float mae_get_cost(float* x, float* y, int length);
void mae_get_cost_derv(float* x, float* y, float* out, int length);

const Cost_func MAE_org = (Cost_func){.get_cost=mae_get_cost, .get_cost_derv=mae_get_cost_derv};
const Cost_func* const MAE = &MAE_org;

/**************** Huber ****************/
float huber_get_cost(float* x, float* y, int length);
void huber_get_cost_derv(float* x, float* y, float* out, int length);

const Cost_func HUBER_org = (Cost_func){.get_cost=huber_get_cost, .get_cost_derv=huber_get_cost_derv};
const Cost_func* const HUBER = &HUBER_org;

#endif