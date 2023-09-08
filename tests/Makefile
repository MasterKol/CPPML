CC=clang++

TEST_CASES= $(shell echo *.cpp)

TEST_LINK_FLAGS=-L/usr/local/opt/libomp/lib -lomp ../libcppml.a -framework Accelerate -I../include -std=c++17 -g

.PHONY: all

all: $(TEST_CASES:.cpp=.tst)
	./runtests

%.tst: %.cpp
	${CC} ${TEST_FLAGS} $< -o $@