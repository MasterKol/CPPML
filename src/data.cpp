#include "data.hpp"
#include <stdio.h>
#include <string>
#include <assert.h>

namespace CPPML {

Data::Data(){
	shape = Shape(0, 0, 0, 0);
	data = NULL;
}

Data::Data(Shape shape_){
	shape = shape_;
	data = new float[shape.size];
}

Data::Data(float* data_, Shape shape_){
	data = data_;
	shape = shape_;
}

void Data::print(std::string frmt){
	printd(data, shape, frmt);
}

float& Data::operator [] (int index){
	return data[index];
}

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

void printd(float* data, Shape shape, std::string frmt){
	for(int d = 0; d < shape.d; d++){
		if(d != 0){
			printf("-------------------------------\n");
		}
		for(int h = 0; h < shape.h; h++){
			for(int w = 0; w < shape.w; w++){
				if(w != 0){
					printf(", ");
				}
				printf(frmt.c_str(), data[(d * shape.h + h) * shape.w + w]);
			}
			printf("\n");
		}
	}
}

void printd(std::string txt, float* data, Shape shape, std::string frmt){
	printf("%s\n", txt.c_str());
	printd(data, shape, frmt);
}

}