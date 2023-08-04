CC=clang++

CFLAGS = ${CFLAGS} -std=c++17 -Xclang -fopenmp -I/usr/local/opt/libomp/include

LIBS = ${LIBS} -L/usr/local/opt/libomp/lib -lomp -lpthread

BP=bin

NORMAL = \
network.o \
layer.o \
helper.o \
data.o \
cost_func.o \
activation.o \
Layers/dense.o \
Layers/conv2d.o \
Layers/maxpooling2d.o \
Layers/upscale2d.o \
Layers/self_attention.o \
Layers/cross_attention.o \
Layers/image_flatten.o \
Layers/input.o \
Optimizers/adam.o \
Optimizers/sgd.o \
diffusion_model.o

OBJECTS = $(addprefix ${BP}/, ${NORMAL})

.PHONY: all

all: ${BP} ${BP}/LinearAlgebra.o ${OBJECTS}

clean:
	rm -r bin

${BP}:
	mkdir ${BP}
	mkdir ${BP}/Layers
	mkdir ${BP}/Optimizers

${BP}/network.o : network.* cost_func.* optimizer.* layer.* data.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

# compiler layers
${BP}/layer.o : layer.* data.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

${BP}/Layers/input.o : Layers/input.* layer.*
	${CC} ${CFLAGS} -c $< -o $@

${BP}/Layers/dense.o : Layers/dense.* layer.* activation.* helper.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

${BP}/Layers/conv2d.o : Layers/conv2d.* layer.* activation.* helper.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

${BP}/Layers/maxpooling2d.o : Layers/maxpooling2d.* layer.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

${BP}/Layers/upscale2d.o : Layers/upscale2d.*p layer.*
	${CC} ${CFLAGS} -c $< -o $@

${BP}/Layers/self_attention.o : Layers/self_attention.* layer.* helper.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

${BP}/Layers/cross_attention.o : Layers/cross_attention.* layer.* helper.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

${BP}/Layers/image_flatten.o : Layers/image_flatten.* layer.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

# compile optimizers
${BP}/adam.o : Optimizers/adam.* optimizer.* network.* LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@

# general compile
${BP}/%.o : %.cpp %.hpp LinearAlgebra.*
	${CC} ${CFLAGS} -c $< -o $@