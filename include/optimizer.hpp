#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cstdio>

#include "network.hpp"

namespace CPPML {

class Network;

class Optimizer {
public:
	Network* net;
	// basically just an alias for the
	// polymorphic function compile_
	void compile(Network* net_){
		net = net_;
		compile_();
	}

	// read params and gradients from network
	// and update the params according to
	// this optimizer's policy
	virtual void update_params() = 0;

	// initialize optimizer (allocate buffers, etc.)
	// net will have already been set
	virtual void compile_() = 0;
};

}

#endif