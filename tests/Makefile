CC=clang++

TEST_CASES= $(shell echo *.cpp)

TEST_FLAGS=-L/usr/local/opt/libomp/lib -lomp ../libcppml.a -framework Accelerate -std=c++17 -Wall -g

#.PHONY: all

#all: $(TEST_CASES:.cpp=.tst)
#	./runtests

%.tst: %.cpp ../libcppml.a layer_test.hpp
	${CC} ${TEST_FLAGS} $< -o $@

clean:
	rm *.tst
	rm -r *.dSYM