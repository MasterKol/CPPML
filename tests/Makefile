CC=clang++

TEST_CASES= $(shell echo *.cpp)

TEST_FLAGS=-L/usr/local/opt/libomp/lib -lomp ../libcppml.a -framework Accelerate -std=c++17

#.PHONY: all

#all: $(TEST_CASES:.cpp=.tst)
#	./runtests

%.tst: %.cpp ../libcppml.a
	${CC} ${TEST_FLAGS} $< -o $@