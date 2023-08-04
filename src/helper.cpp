#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <sys/time.h>
#include <random>
#include "LinearAlgebra.hpp"

/* copied from this stack overflow:
 * https://stackoverflow.com/questions/23369503/get-size-of-terminal-window-rows-columns
 */
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__) || __APPLE__
#include <sys/ioctl.h>
#endif // Windows/Linux/Mac

int get_terminal_width(){
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return (int)(csbi.srWindow.Right-csbi.srWindow.Left+1);
#elif defined(__linux__) || __APPLE__
    struct winsize w;
    ioctl(fileno(stdout), TIOCGWINSZ, &w);
    return (int)(w.ws_col);
#endif // Windows/Linux/Mac
}
/*end copied section*/

void get_sin_pos_embed(int v, float* emb, int d, float n){
	assert(d % 2 == 0); // d must be even

	const float W = powf(10000.0f, -2.0f/d);
	float wk = W;
	for(int k = 0; k < d; k+=2){
		emb[k  ] = sinf(wk * v);
		emb[k+1] = cosf(wk * v);
		wk *= W;
	}
}

long micro(){
	static struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + (long)tv.tv_usec;
}

std::mt19937 rng(0);
std::uniform_real_distribution<float> unif_dist(0.0f, 1.0f);
std::normal_distribution<float> norm_dist{0, 1};

int randI(int max){
	return std::floor(unif_dist(rng) * max);
}

int randI(int min, int max){
	return std::floor(min + unif_dist(rng) * (max - min));
}

float randF(float min, float max){
	return min + unif_dist(rng) * (max - min);
}

void fillRand(float* a, int N, float min, float max){
	for(int i = 0; i < N; i++){
		a[i] = unif_dist(rng);
	}
	float m = (max - min);
	vDSP_vsmsa(a, 1, &m, &min, a, 1, N);
}

float randomGaussian(float mean, float sdv){
	return norm_dist(rng) * sdv + mean;
}

void fillGaussian(float* a, int N, float mean, float sdv){
	for(int i = 0; i < N; i++){
		a[i] = norm_dist(rng);
	}
	vDSP_vsmsa(a, 1, &sdv, &mean, a, 1, N);
}

void rand_seed(int seed){
	rng = std::mt19937(seed);
}