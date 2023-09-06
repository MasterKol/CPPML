#include "shape.hpp"

#include <iostream>
#include <string>
#include <cassert>

namespace CPPML {

Shape::Shape(int _w_, int _h_, int _d_, int _n_){
	w_ = _w_;
	h_ = _h_;
	d_ = _d_;
	n_ = _n_;
	size_ = w_ * h_ * d_ * n_;
}

int Shape::operator [] (int ind){
	return ((int*)this)[ind];
}

void Shape::fix_size(){
	size_ = w_ * h_ * d_ * n_;
}

void Shape::print(){
	std::cout << to_string() << std::endl;
}

std::string Shape::to_string(){
	std::string out = "(";
	for(int i = 0; i < 3; i++){
		out += std::to_string((*this)[i]) + ", ";
	}
	out += std::to_string(n_) + ")";
	return out;
}

template<typename... args>
std::string format(std::string format_string, args... args1){
	// get size of final string
	int buff_size = snprintf(nullptr, 0, format_string.c_str(), args1...);

	// make string of output size
	std::string out;
	out.resize(buff_size);
	
	// write string to given size
	snprintf(&out[0], buff_size, format_string.c_str(), args1...);

	// remove null terminator
	if(out.size() != 0)
		out.pop_back();

	return out;
}

void Shape::printd(float* data, std::string frmt){
	for(int di = 0; di < d_; di++){
		if(di != 0){
			std::cout << "-------------------------------\n";
		}
		for(int hi = 0; hi < h_; hi++){
			for(int wi = 0; wi < w_; wi++){
				if(wi != 0)
					std::cout << ", ";
				std::cout << format(frmt, data[(di * h_ + hi) * w_ + wi]);
			}
			std::cout << "\n";
		}
	}
	std::cout << std::flush;
}

void Shape::printd(std::string txt, float* data, std::string frmt){
	std::cout << txt << std::endl;
	printd(data, frmt);
}

int Shape::size(){
	if(size_ == -1)
		size_ = w_ * h_ * d_ * n_;
	return size_;
}

void Shape::w(int new_w){
	w_ = new_w;
	size_ = -1;
}

void Shape::h(int new_h){
	h_ = new_h;
	size_ = -1;
}

void Shape::d(int new_d){
	d_ = new_d;
	size_ = -1;
}

void Shape::n(int new_n){
	n_ = new_n;
	size_ = -1;
}

} // namespace CPPML