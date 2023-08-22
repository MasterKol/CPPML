#include "data.hpp"
#include <iostream>
#include <string>
#include <assert.h>

namespace CPPML {

Shape operator + (Shape lhs, Shape rhs){
	Shape out = lhs;

	int i = 0;
	for(; i < 4; i++){
		if(lhs[i] != rhs[i]){
			out[i] += rhs[i];
			i++;
			break;
		}
	}

	for(; i < 4; i++){
		// dimensions do not match
		assert(lhs[i] == rhs[i]);
	}

	out.size = lhs.size + rhs.size;

	return out;
}

int& Shape::operator [] (int ind){
	return ((int*)this)[ind];
}

void Shape::fix_size(){
	size = w * h * d * n;
}

void Shape::print(){
	std::cout << to_string() << std::endl;
}

std::string Shape::to_string(){
	std::string out = "(";
	for(int i = 0; i < 3; i++){
		out += std::to_string((*this)[i]) + ", ";
	}
	out += std::to_string(n) + ")";
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
	for(int di = 0; di < d; di++){
		if(di != 0){
			std::cout << "-------------------------------\n";
		}
		for(int hi = 0; hi < h; hi++){
			for(int wi = 0; wi < w; wi++){
				if(wi != 0)
					std::cout << ", ";
				std::cout << format(frmt, data[(di * h + hi) * w + wi]);
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

}