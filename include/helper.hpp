#ifndef HELPER_H
#define HELPER_H

#include <stdlib.h>
#include <cmath>

namespace CPPML {

long micro();
float randF(float min, float max);

void fillRand(float* a, int N, float min, float max);

float randomGaussian(float mean, float sdv);

void fillGaussian(float* a, int N, float mean, float sdv);

int randI(int min, int max);
int randI(int max);
void rand_seed(int seed);

int get_terminal_width();

void get_sin_pos_embed(int v, float* emb, int d, float n=10000);

}

#endif