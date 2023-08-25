#ifndef DATA_H
#define DATA_H

#include <string>

namespace CPPML {

/*
 * Defines the shape of a vector of data
 * w changes first, then h, then d, then n
 */
class Shape{
private:
	int w_, h_, d_, n_;
	int size_;
public:
	Shape(int w = 1, int h = 1, int d = 1, int n = 1);

	// add two shapes together along largest non-one axis
	// throws an error if lower axes don't match
	//friend Shape operator + (Shape lhs, Shape rhs);

	// allows array like accessing of info
	// 0 = w, 1 = h, 2 = d, 3 = n
	int operator [] (int ind);

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

	int w() const { return w_; }
	int h() const { return h_; }
	int d() const { return d_; }
	int n() const { return n_; }
	int size();

	void w(int new_w);
	void h(int new_h);
	void d(int new_d);
	void n(int new_n);

private:
	// if any sizes are changed then fix_size must
	void fix_size();
};

}

#endif