#ifndef DATA_H
#define DATA_H

#include <string>

namespace CPPML {

/*
 * Defines the shape of a vector of data
 * w changes first, then h, then d, then n
 */
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

	// add two shapes together along largest non-one axis
	// throws an error if lower axes don't match
	friend Shape operator + (Shape lhs, Shape rhs);

	// allows array like accessing of info
	// 0 = w, 1 = h, 2 = d, 3 = n
	int& operator [] (int ind);

	// if any sizes are changed then fix_size must
	void fix_size();

	// prints shape to stdout
	void print();

	// returns Shape as string in the format
	// (w, h, d, n)
	std::string to_string();

	// prints the given data formatted for this shape
	// optionally accepts format specifier for changing output
	void printd(float* data, std::string frmt="% .3f");

	// prints the given data formatted for this shape
	// optionally accepts format specifier for changing output
	void printd(std::string txt, float* data, std::string frmt="% .3f");
};

}

#endif