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
activation_func.o \
LinearAlgebra.o

OBJECTS = $(addprefix ${BP}/, ${NORMAL})

.PHONY: all test

all: src/Makefile
#	remake libcppml.a only if other make file needs to do something
	(cd src && make -q || (make && cd .. && ar -r libcppml.a ${OBJECTS}))

src/Makefile: src/*.* src/*/*.* include/*.* include/*/*.*
	bash remake.sh

clean:
	rm bin/*.o || true
	rm src/Makefile || true
	rm tests/*.tst || true
	rm -r tests/*.dSYM || true

test: all
	./tests/runtests