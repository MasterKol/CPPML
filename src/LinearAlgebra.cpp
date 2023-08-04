#include "LinearAlgebra.hpp"

#if !defined(ACCEL)//!(defined(__has_include) && __has_include(<Accelerate/Accelerate.h>))

#include <cmath>
#include <string.h>

void vDSP_vfill(const float* v, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = *v;
		out += OutStride;
	}
}

void vDSP_vnabs(const float* in, int InStride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = -abs(*in);
		in += InStride;
		out += OutStride;
	}
}

void vDSP_vmax(const float* A, int Astride, const float* B, int Bstride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = fmax(*A, *B);
		out += OutStride;
		A += Astride;
		B += Bstride;
	}
}

void vvexpf(float* out, const float* in, const int* N){
	for(int i = 0; i < *N; i++){
		out[i] = expf(in[i]);
	}
}

void vDSP_vclip(const float* in, int InStride, const float* low, const float* high, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		if(*in <= *low){
			*out = *low;
		}else if(*in >= *high){
			*out = *high;
		}else{
			*out = *in;
		}
		in += InStride;
		out += OutStride;
	}
}

void vDSP_vthres(const float* in, int InStride, const float* B, float *out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = (*in < *B ? 0 : *in);
		in += InStride;
		out += OutStride;
	}
}

void vDSP_vthrsc(const float* in, int InStride, const float* B, const float* C, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = (*in > *B) ? *C : -*C;
		in += InStride;
		out += OutStride;
	}
}

void vDSP_vneg(const float* in, int InStride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = -*in;
		in += InStride;
		out += OutStride;
	}
}

void vDSP_vsadd(const float* in, int InStride, const float* v, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = *in + *v;
		in += InStride;
		out += OutStride;
	}
}

void vvrecf(float* out, const float* in, const int* N){
	for(int i = 0; i < *N; i++){
		*out = 1.0f / *in;
		in++;
		out++;
	}
}

void vDSP_vmsb(const float* A, int Astride, const float* B, int Bstride, const float* C, int Cstride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = (*A) * (*B) - *C;
		A += Astride;
		B += Bstride;
		C += Cstride;
		out += OutStride;
	}
}

void vDSP_distancesq(const float* A, int Astride, const float* B, int Bstride, float* out, int N){
	*out = 0;
	for(int i = 0; i < N; i++){
		float t = *A - *B;
		*out += t * t;
		A += Astride;
		B += Bstride;
	}
}

void vDSP_vsbsm(const float *A, int Astride, const float* B, int Bstride, const float* C, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = (*A - *B) * (*C);
		A += Astride;
		B += Bstride;
		out += OutStride;
	}
}

void vDSP_vsub(const float* B, int Bstride, const float* A, int Astride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = *A - *B;
		A += Astride;
		B += Bstride;
		out += OutStride;
	}
}

void vDSP_vadd(const float* A, int Astride, const float *B, int Bstride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = *A + *B;
		A += Astride;
		B += Bstride;
		out += OutStride;
	}
}

void vvcopysignf(float* out, const float* A, const float* B, const int* N){
	for(int i = 0; i < *N; i++){
		out[i] = copysignf(A[i], B[i]);
	}
}

void vDSP_vsmul(const float* in, int InStride, const float* v, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = (*in) * (*v);
		in += InStride;
		out += OutStride;
	}
}

void vDSP_sve(const float* in, int InStride, float* out, int N){
	*out = 0;
	for(int i = 0; i < N; i++){
		*out += *in;
		in += InStride;
	}
}

void vDSP_dotpr(const float* A, int Astride, const float* B, int Bstride, float* out, int N){
	*out = 0;
	for(int i = 0; i < N; i++){
		*out += (*A) * (*B);
		A += Astride;
		B += Bstride;
	}
}

void vDSP_vintb(const float* A, int Astride, const float* B, int Bstride, const float* t, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = *A + (*t) * (*B - *A);
		A += Astride;
		B += Bstride;
		out += OutStride;
	}
}

void vDSP_vsq(const float* in, int InStride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = (*in) * (*in);
		in += InStride;
		out += OutStride;
	}
}

void vvsqrtf(float* out, const float* in, const int* N){
	for(int i = 0; i < *N; i++){
		out[i] = sqrtf(in[i]);
	}
}

void vDSP_vsmsa(const float* in, int InStride, const float* A, const float* B, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = (*in) * (*A) + (*B);
		in += InStride;
		out += OutStride;
	}
}

void vDSP_vdiv(const float* B, int Bstride, const float* A, int Astride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = *A / *B;
		A += Astride;
		B += Bstride;
		out += OutStride;
	}
}

void vDSP_vmul(const float* A, int Astride, const float* B, int Bstride, float* out, int OutStride, int N){
	for(int i = 0; i < N; i++){
		*out = (*A) * (*B);
		A += Astride;
		B += Bstride;
		out += OutStride;
	}
}

void vvexpm1f(float* out, const float* in, const int* N){
	for(int i = 0; i < *N; i++){
		*out = expf(*in) - 1.0f;
		in++;
		out++;
	}
}

void vDSP_svemg(const float* in, int InStride, float* out, int N){
	*out = 0;
	for(int i = 0; i < N; i++){
		*out += abs(*in);
		in += InStride;
	}
}

void vDSP_mmul(const float* A, int Astride, const float* B, int Bstride, float* out, int OutStride, int M, int N, int P){
	int Aroff = Astride * P;
	int Broff = Bstride * N;
	
	const float* Bcol = B;
	for(int ro = 0; ro < M; ro++){ // loop over output rows
		for(int co = 0; co < N; co++){ // loop over output columns
			*out = 0;

			// write dot product between row of A and column of B to out
			vDSP_dotpr(A, Astride, Bcol, Broff, out, P);
			
			Bcol += Bstride;
			out += OutStride;
		}
		A += Aroff;
		Bcol = B;
	}
}

void mmul_transpose(const int M, const int N, const float alpha, const float* A, const int Astride, const float* B, const int Bstride, float* out){
	const float* Bstart = B;
	for(int ro = 0; ro < M; ro++){ // loop over output rows
		for(int co = 0; co < N; co++){ // loop over output columns
			*out += (*A) * (*B);
			B += Bstride;
			out++;
		}
		A += Astride;
		B = Bstart;
	}
}

void vDSP_vsma(const float *A, const int Astride, const float *B, const float *C, const int Cstride, float *out, int OutStride, const int N){
	for(int i = 0; i < N; i++){
		*out = *A * (*B) + *C;

		A += Astride;
		C += Cstride;
		out += OutStride;
	}
}

void vDSP_mmov(const float *src, float *dst, int cols, int rows, int SrcCols, int DstCols){
	for(int r = 0; r < rows; r++){
		memcpy(dst, src, cols * sizeof(float));
		src += SrcCols;
		dst += DstCols;
	}
}

void vDSP_vsdiv(const float *in, int InStride, const float *D, float *out, int OutStride, int N){
	// this technically isn't 100% correct but it should basically work
	float denom = 1.0f / *D;
	vDSP_vsmul(in, InStride, &denom, out, OutStride, N);
}

void vDSP_maxv(const float *in, int InStride, float *out, int N){
	*out = *in;
	for(int i = 1; i < N; i++){
		in += InStride;
		if(*out < *in){
			*out = *in;
		}
	}
}

void vDSP_mtrans(const float *in, int InStride, float *out, int OutStride, int ORows, int OCols){
	const int ORow_offset = OutStride * OCols;
	for(int i = 0; i < OCols; i++){
		for(int j = 0; j < ORows; j++){
			out[j * ORow_offset] = *in;
			in += InStride;
		}

		out += OutStride;
	}
}

void vDSP_vsmsma(const float *A, int AStride, const float *B, const float *C, int CStride, const float *D, float *E, int EStride, int N){
	for(int i = 0; i < N; i++){
		*E = *A * (*B) + *C + (*D);

		A += AStride;
		C += CStride;
		E += EStride;
	}
}

void vDSP_vma(const float *A, int AStride, const float *B, int BStride, const float *C, int CStride, float *D, int DStride, int N){
	for(int i = 0; i < N; i++){
		*D = *A * (*B) + *C;

		A += AStride;
		B += BStride;
		C += CStride;
		D += DStride;
	}
}

void vDSP_vmmsb(const float *A, int AStride, const float *B, int BStride, const float *C, int CStride, const float *D, int DStride, float *E, int EStride, int N){
	for(int i = 0; i < N; i++){
		*E = *A * (*B) - *C * (*D);

		A += AStride;
		B += BStride;
		C += CStride;
		D += DStride;
		E += EStride;
	}
}

#endif