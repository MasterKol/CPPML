#include "random.hpp"

#include <cmath>
#include <random>
#include <chrono>

#include "LinearAlgebra.hpp"

namespace CPPML {

std::mt19937 Random::rng = std::mt19937(0);

int Random::randI(int max){
	std::uniform_int_distribution<int> distribution (0, max-1);
	return distribution(rng);
}

int Random::randI(int min, int max){
	std::uniform_int_distribution<int> distribution (min, max-1);
	return distribution(rng);
}

float Random::randF(float min, float max){
	std::uniform_real_distribution<float> distribution (min, max);
	return distribution(rng);
}

void Random::fillRand(float* a, int N, float min, float max){
	std::uniform_real_distribution<float> distribution (min, max);
	for(int i = 0; i < N; i++){
		a[i] = distribution(rng);
	}
}

float Random::randomGaussian(float mean, float sdv){
	std::normal_distribution<float> distribution (mean, sdv);
	return distribution(rng);
}

void Random::fillGaussian(float* a, int N, float mean, float sdv){
	std::normal_distribution<float> distribution (mean, sdv);
	for(int i = 0; i < N; i++){
		a[i] = distribution(rng);
	}
}

void Random::rand_seed(int seed){
	rng = std::mt19937(seed);
}

int Random::time_seed(){
	using namespace std::chrono;
	int ms = duration_cast< nanoseconds >(
		steady_clock::now().time_since_epoch()
	).count();
	rng = std::mt19937(ms);
	return ms;
}

} // namespace CPPML