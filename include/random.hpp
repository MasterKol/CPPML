#ifndef HELPER_H
#define HELPER_H

#include <random>

namespace CPPML {

/*
 * Handles generation of random numbers
*/
class Random {
private:
	static std::mt19937 rng;
	static std::uniform_real_distribution<float> unif_dist;
	static std::normal_distribution<float> norm_dist;
public:

	// returns uniform random float in the range (min, max)
	static float randF(float min, float max);

	// fills array of length N with uniform random floats
	// in the range (min, max)
	static void fillRand(float* a, int N, float min, float max);

	// returns random gaussian float with given mean and standard deviation
	static float randomGaussian(float mean, float sdv);

	// fills array of length N with gaussian floats with given mean and standard deviation
	static void fillGaussian(float* a, int N, float mean, float sdv);

	// returns random int in the range (min, max)
	static int randI(int min, int max);

	// returns random int in the range (0, max)
	static int randI(int max);

	// sets the random seed
	static void rand_seed(int seed);
};

}

#endif