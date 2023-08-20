#ifndef DATA_H
#define DATA_H

#include <string>

namespace CPPML {

class Shape{
public:
	int w, h, d, n;
	int size;

	Shape(int w_ = 1, int h_ = 1, int d_ = 1, int n_ = 1){
		w = w_;
		h = h_;
		d = d_;
		n = n_;
		size = w * h * d * n;
	}

	friend Shape operator + (Shape lhs, Shape rhs);
	int& operator [] (int ind){
		return ((int*)this)[ind];
	}

	void fix_size(){
		size = w * h * d * n;
	}

	void print(){
		printf("Shape: [%d, %d, %d, %d]\n", w, h, d, n);
	}

	std::string to_string(){
		std::string out = "(";
		for(int i = 0; i < 3; i++){
			out += std::to_string((*this)[i]) + ", ";
		}
		out += std::to_string(n) + ")";
		return out;
	}
};

class Data {
public:
	float* data;
	Shape shape;

	Data(float*, Shape);
	Data(Shape);
	Data();

	void print(std::string frmt="%.6d");
	float& operator [] (int);
};

void printd(float* data, Shape shape, std::string frmt="% .3f");
void printd(std::string txt, float* data, Shape shape, std::string frmt="% .3f");

}

#endif