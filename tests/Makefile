CC:=clang++

TEST_CASES:=$(shell echo *.cpp)

TEST_FLAGS:=-L/usr/local/opt/libomp/lib -I../include -lomp ../libcppml.a -framework Accelerate -std=c++17 -Wall

dependencies=../libcppml.a

.PHONY=all

all:
	./runtests.py

# for all files
layer_tests/*/*.tst: dependencies+=layer_tests/layer_test.hpp

%.tst: %.cpp ${dependencies}
#	echo 456 ${dependencies}
	${CC} ${CXXFLAGS} ${TEST_FLAGS} $< -o $@

clean:
	./clean.py