 
CC=clang++
BP=../bin
CFLAGS=-std=c++17 -Xclang -fopenmp -I/usr/local/opt/libomp/include -I../include
INC=../include
.PHONY: all
all: ${BP} ${BP}/LinearAlgebra.o ${BP}/activation.o ${BP}/cost_func.o ${BP}/data.o ${BP}/helper.o ${BP}/layer.o ${BP}/network.o ${BP}/conv2d.o ${BP}/cross_attention.o ${BP}/dense.o ${BP}/image_flatten.o ${BP}/input.o ${BP}/maxpooling2d.o ${BP}/self_attention.o ${BP}/upscale2d.o ${BP}/adam.o ${BP}/sgd.o 

${BP}:
	mkdir ${BP}
	mkdir ${BP}/Layers
	mkdir ${BP}/Optimizers

${BP}/LinearAlgebra.o : ${INC}/./LinearAlgebra.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/activation.o : ${INC}/./activation.hpp ${INC}/./LinearAlgebra.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/cost_func.o : ${INC}/./cost_func.hpp ${INC}/./LinearAlgebra.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/data.o : ${INC}/./data.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/helper.o : ${INC}/./LinearAlgebra.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/layer.o : ${INC}/./layer.hpp ${INC}/./data.hpp ${INC}/./LinearAlgebra.hpp ${INC}/./data.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/network.o : ${INC}/./network.hpp ${INC}/./LinearAlgebra.hpp ${INC}/./helper.hpp ${INC}/./optimizer.hpp ${INC}/./cost_func.hpp ${INC}/./optimizer.hpp ${INC}/./layer.hpp ${INC}/./data.hpp ${INC}/./Layers/input.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/conv2d.o : ${INC}/./Layers/conv2d.hpp ${INC}/./Layers/../activation.hpp ${INC}/./Layers/../helper.hpp ${INC}/./Layers/../LinearAlgebra.hpp ${INC}/./Layers/../activation.hpp ${INC}/./Layers/../layer.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/cross_attention.o : ${INC}/./Layers/cross_attention.hpp ${INC}/./Layers/../LinearAlgebra.hpp ${INC}/./Layers/../helper.hpp ${INC}/./Layers/../layer.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/dense.o : ${INC}/./Layers/dense.hpp ${INC}/./Layers/../helper.hpp ${INC}/./Layers/../LinearAlgebra.hpp ${INC}/./Layers/../layer.hpp ${INC}/./Layers/../activation.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/image_flatten.o : ${INC}/./Layers/image_flatten.hpp ${INC}/./Layers/../LinearAlgebra.hpp ${INC}/./Layers/../layer.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/input.o : ${INC}/./Layers/input.hpp ${INC}/./Layers/../layer.hpp ${INC}/./Layers/../layer.hpp ${INC}/./Layers/../network.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/maxpooling2d.o : ${INC}/./Layers/maxpooling2d.hpp ${INC}/./Layers/../LinearAlgebra.hpp ${INC}/./Layers/../layer.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/self_attention.o : ${INC}/./Layers/self_attention.hpp ${INC}/./Layers/../helper.hpp ${INC}/./Layers/../LinearAlgebra.hpp ${INC}/./Layers/../layer.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/upscale2d.o : ${INC}/./Layers/upscale2d.hpp ${INC}/./Layers/../layer.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/adam.o : ${INC}/./Optimizers/adam.hpp ${INC}/./Optimizers/../LinearAlgebra.hpp ${INC}/./Optimizers/../data.hpp ${INC}/./Optimizers/../optimizer.hpp ${INC}/./Optimizers/../network.hpp
	${CC} ${CFLAGS} -c $< -o $@

${BP}/sgd.o : ${INC}/./Optimizers/sgd.hpp ${INC}/./Optimizers/../LinearAlgebra.hpp ${INC}/./Optimizers/../optimizer.hpp ${INC}/./Optimizers/../network.hpp
	${CC} ${CFLAGS} -c $< -o $@

