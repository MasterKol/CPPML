CC:=clang++

TEST_CASES:=$(shell echo *.cpp)

TEST_FLAGS:=-L/usr/local/opt/libomp/lib -I../include -lomp ../libcppml.a -framework Accelerate -std=c++17 -Wall

dependencies:=../libcppml.a

# for all files
layer_tests/%.tst: dependencies+=layer_tests/layer_test.hpp

%.tst: %.cpp ${dependencies}
	${CC} ${CXXFLAGS} ${TEST_FLAGS} $< -o $@

%.123:
	echo ${CXXFLAGS}

clean:
	./clean.py