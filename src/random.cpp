#include "random.hpp"

#include <cmath>
#include <random>
#include <chrono>

#include "LinearAlgebra.hpp"

namespace CPPML {

std::mt19937 Random::rng = std::mt19937(0);
std::uniform_real_distribution<float> Random::unif_dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
std::normal_distribution<float> Random::norm_dist = std::normal_distribution<float>{0, 1};

int Random::randI(int max){
	return std::floor(unif_dist(rng) * max);
}

int Random::randI(int min, int max){
	return std::floor(min + unif_dist(rng) * (max - min));
}

float Random::randF(float min, float max){
	return min + unif_dist(rng) * (max - min);
}

void Random::fillRand(float* a, int N, float min, float max){
	for(int i = 0; i < N; i++){
		a[i] = unif_dist(rng);
	}
	float m = (max - min);
	vDSP_vsmsa(a, 1, &m, &min, a, 1, N);
}

float Random::randomGaussian(float mean, float sdv){
	return norm_dist(rng) * sdv + mean;
}

void Random::fillGaussian(float* a, int N, float mean, float sdv){
	for(int i = 0; i < N; i++){
		a[i] = norm_dist(rng);
	}
	vDSP_vsmsa(a, 1, &sdv, &mean, a, 1, N);
}

void Random::rand_seed(int seed){
	rng = std::mt19937(seed);
}

int Random::time_seed(){
	using namespace std::chrono;
	int ms = duration_cast< milliseconds >(
		steady_clock::now().time_since_epoch()
	).count();
	rng = std::mt19937(ms);
	return ms;
}

} // namespace CPPML