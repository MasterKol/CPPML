#ifndef HELPER_H
#define HELPER_H

#include <random>

namespace CPPML {

/*
 * Handles generation of random numbers
 * Not thread safe
*/
class Random {
public:
	static std::mt19937 rng;
	
	/// @brief returns uniform random float in the range (min, max)
	/// @param min minimum possible value
	/// @param max maximum possible value
	static float randF(float min, float max);

	/// @brief fills array of length N with uniform random floats in the range (min, max)
	/// @param array array to fill
	/// @param N length of array
	/// @param min minimum possible value
	/// @param max maximum possible value
	static void fillRand(float* array, int N, float min, float max);

	/// @brief returns random gaussian float with given mean and standard deviation
	/// @param mean mean of the distribution
	/// @param sdv standard deviation of the distribution
	static float randomGaussian(float mean, float sdv);

	/// @brief fills array of length N with gaussian floats with given mean and standard deviation
	/// @param array array to fill
	/// @param N length of array
	/// @param mean mean of the distribution
	/// @param sdv standard deviation of the distribution
	static void fillGaussian(float* array, int N, float mean, float sdv);

	/// @brief returns random int in the range (min, max). Not great for ranges > ~16777216
	/// @param min minimum possible value (inclusive)
	/// @param max maximum possible value (exclusive)
	static int randI(int min, int max);


	/// @brief returns random int in the range (0, max). Not great for ranges > ~16777216
	/// @param max maximum possible value (exclusive)
	static int randI(int max);

	/// @brief sets the random seed
	static void rand_seed(int seed);

	/// @brief sets the random seed based on the current time in nanoseconds
	/// @return the seed that was used
	static int time_seed();
};

}

#endif