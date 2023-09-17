BP=bin

NORMAL = \
network.o \
layer.o \
random.o \
shape.o \
cost_func.o \
activation.o \
dense.o \
conv2d.o \
maxpooling2d.o \
upscale2d.o \
self_attention.o \
cross_attention.o \
image_flatten.o \
input.o \
adam.o \
sgd.o \
LinearAlgebra.o

OBJECTS = $(addprefix ${BP}/, ${NORMAL})

.PHONY: all test

all:
	bash remake.sh
#	remake libcppml.a only if other make file needs to do something
	(cd src && make -q || (make && cd .. && ar -r libcppml.a ${OBJECTS}))

clean:
	rm bin/*.o
	rm src/Makefile
	rm tests/*.tst

test: all
	./tests/runtests