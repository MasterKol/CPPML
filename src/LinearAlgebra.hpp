#ifndef LINEAR_ALGEBRA_HEADER
#define LINEAR_ALGEBRA_HEADER
/* 
 * This file manages the linear algebra libraries used in this 
 * project and (hopefully) allows for it to run cross platform.
 * These will all just be redirects to other functions so they
 * can be inlined. If no good library can be determined (or I 
 * haven't added support for it) I wrote some fallback code so 
 * that this will hopefully work on every platform. I tried to
 * stick as close as possible to the signature that Accelerate
 * has for these functions but I changed some for convenience.
 */

// vDSP_vfill 		(const float *__A, float *__C, vDSP_Stride __IC, vDSP_Length __N)
// vDSP_vnabs 		(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vDSP_vmax 		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vvexpf 			(float *, const float *, const int *);
// vDSP_vclip		(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_vthres		(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vDSP_vthrsc		(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_vneg		(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vDSP_vsadd		(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vvrecf			(float *, const float *, const int *);
// vDSP_vmsb		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, const float *__C, vDSP_Stride __IC, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_distancesq	(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C, vDSP_Length __N);
// vDSP_vsbsm		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, const float *__C, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_vsub		(const float *__B, vDSP_Stride __IB, const float *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vDSP_vadd		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vvcopysignf		(float *, const float *, const float *, const int *);
// vDSP_vsmul		(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vDSP_vsma		(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, vDSP_Stride __IC, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_sve			(const float *__A, vDSP_Stride __I, float *__C, vDSP_Length __N);
// vDSP_dotpr		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C, vDSP_Length __N);
// vDSP_vintb		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, const float *__C, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_vsq			(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vvsqrtf			(float *, const float *, const int *);
// vDSP_vsmsa		(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_vdiv		(const float *__B, vDSP_Stride __IB, const float *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vDSP_vmul		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vvexpm1f			(float *, const float *, const int *);
// vDSP_svemg		(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length __N);
// vDSP_mmul		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C, vDSP_Stride __IC, vDSP_Length __M, vDSP_Length __N, vDSP_Length __P);
// cblas_sger		(const enum CBLAS_ORDER ORDER, const __LAPACK_int M, const __LAPACK_int N, const float ALPHA, const float *X, const __LAPACK_int INCX, const float *Y, const __LAPACK_int INCY, float *A, const __LAPACK_int LDA);
// vDSP_mmov		(const float *__A, float *__C, vDSP_Length __M, vDSP_Length __N, vDSP_Length __TA, vDSP_Length __TC);
// vDSP_vsdiv		(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC, vDSP_Length __N);
// vDSP_maxv		(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length __N);
// vDSP_mtrans		(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __M, vDSP_Length __N);
// vDSP_vsmsma		(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, vDSP_Stride __IC, const float *__D, float *__E, vDSP_Stride __IE, vDSP_Length __N);
// vDSP_vmsb		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, const float *__C, vDSP_Stride __IC, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_vma			(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, const float *__C, vDSP_Stride __IC, float *__D, vDSP_Stride __ID, vDSP_Length __N);
// vDSP_vmmsb		(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, const float *__C, vDSP_Stride __IC, const float *__D, vDSP_Stride __ID, float *__E, vDSP_Stride __IE, vDSP_Length __N);

// Has the accelerate framework (OSX only I think)
//#define ACCEL

#if defined(__has_include) && __has_include(<Accelerate/Accelerate.h>)
	#include <Accelerate/Accelerate.h>
#else
	namespace CPPML {
	// fills array out with value v
	void vDSP_vfill(const float* v, float* out, int OutStride, int N);
	// writes negative abs of vector in to out
	void vDSP_vnabs(const float* in, int InStride, float* out, int OutStride, int N);
	// calculates the maximum of a and b and writes it to out
	void vDSP_vmax(const float* A, int Astride, const float* B, int Bstride, float* out, int OutStride, int N);
	// calculates out <- e^in
	void vvexpf(float* out, const float* in, const int* N);
	// calculates out <- constrain(in, low, high)
	void vDSP_vclip(const float* in, int InStride, const float* low, const float* high, float* out, int OutStride, int N);
	// writes to out, if in is less than B than set to 0 other wise keep in
	void vDSP_vthres(const float* in, int InStride, const float* B, float *out, int OutStride, int N);
	// if in is greater than B than write +C otherwise write -C
	void vDSP_vthrsc(const float* in, int InStride, const float* B, const float* C, float* out, int OutStride, int N);
	// calculates negative of input
	void vDSP_vneg(const float* in, int InStride, float* out, int OutStride, int N);
	// adds a scalar to a vector
	void vDSP_vsadd(const float* in, int InStride, const float* v, float* out, int OutStride, int N);
	// Calculates the reciprocal of each element in an array of single-precision values.
	void vvrecf(float* out, const float* in, const int* N);
	// Subtracts a single-precision vector from the product of two single-precision vectors (out = A * B - C)
	void vDSP_vmsb(const float* A, int Astride, const float* B, int Bstride, const float* C, int Cstride, float* out, int OutStride, int N);
	// Calculates the distance squared between two single-precision points in n-dimensional space (out = sum(in * in))
	void vDSP_distancesq(const float* A, int Astride, const float* B, int Bstride, float* out, int N);
	// Multiplies the difference of two single-precision vectors by a single-precision scalar value. (out = (A - B) * C)
	void vDSP_vsbsm(const float *A, int Astride, const float* B, int Bstride, const float* C, float* out, int OutStride, int N);
	// Subtracts two single-precision vectors. (out = A - B)
	void vDSP_vsub(const float* B, int Bstride, const float* A, int Astride, float* out, int OutStride, int N);
	// Adds two single-precision vectors. (out = A + B)
	void vDSP_vadd(const float* A, int Astride, const float *B, int Bstride, float* out, int OutStride, int N);
	// Copies an array, setting the sign of each element based on a second array of single-precision values (out = abs(A) * sign(B))
	void vvcopysignf(float* out, const float* A, const float* B, const int* N);
	// Multiplies a single-precision scalar value by a single-precision vector. (out = in * v)
	void vDSP_vsmul(const float* in, int InStride, const float* v, float* out, int OutStride, int N);
	// Calculates the sum of values in a single-precision vector. (out = sum(A))
	void vDSP_sve(const float* in, int InStride, float* out, int N);
	// Calculates the dot product of a single-precision vector. (out = sum(A * B))
	void vDSP_dotpr(const float* A, int Astride, const float* B, int Bstride, float* out, int N);
	// Calculates the linear interpolation between the supplied single-precision vectors using the specified stride. (out = A + t * (B - A))
	void vDSP_vintb(const float* A, int Astride, const float* B, int Bstride, const float* t, float* out, int OutStride, int N);
	// Computes the squared value of each element in the supplied single-precision vector. (out = in * in)
	void vDSP_vsq(const float* in, int InStride, float* out, int OutStride, int N);
	// Calculates the square root of each element in an array of single-precision values. (out = sqrt(x))
	void vvsqrtf(float* out, const float* in, const int* N);
	// Adds a single-precision scalar value to the product of a single-precision vector and a single-precision scalar value. (out = in * B + C)
	void vDSP_vsmsa(const float* in, int InStride, const float* A, const float* B, float* out, int OutStride, int N);
	// Multiplies two single-precision vectors. (out = A * B)
	void vDSP_vmul(const float* A, int Astride, const float* B, int Bstride, float* out, int OutStride, int N);
	// Calculates e^x-1 for each element in an array of single-precision values. (out = (e^in) - 1)
	void vvexpm1f(float* out, const float* in, const int* N);
	// Calculates the sum of magnitudes in a single-precision vector (out = sum(abs(in)))
	void vDSP_svemg(const float* in, int InStride, float* out, int N);
	// Divides two single-precision vectors.
	void vDSP_vdiv(const float* B, int Bstride, const float* A, int Astride, float* out, int OutStride, int N);
	// Computes the matrix operation out = A * B, matricies are of sizes (row, colum) out = (M, N), A = (M, P), B = (P, N)
	void vDSP_mmul(const float* A, int Astride, const float* B, int Bstride, float* out, int OutStride, int M, int N, int P);
	// Adds a single-precision vector to the product of a single-precision scalar value and a single-precision vector.
	void vDSP_vsma(const float *A, const int Astride, const float *B, const float *C, const int Cstride, float *out, int OutStride, const int N);
	// Copies the contents of a submatrix to another submatrix; single precision.
	void vDSP_mmov(const float *src, float *dst, int cols, int rows, int SrcCols, int DstCols);
	// Divides a single-precision vector by a single-precision scalar value.
	void vDSP_vsdiv(const float *in, int InStride, const float *D, float *out, int OutStride, int N);
	// Calculates the maximum value in a single-precision vector.
	void vDSP_maxv(const float *in, int InStride, float *out, int N);
	// Creates a transposed matrix C from a source matrix A; single precision. OUT OF PLACE
	void vDSP_mtrans(const float *in, int InStride, float *out, int OutStride, int ORows, int OCols);
	// Adds the product of a single-precision vector and a single-precision scalar value to a second product of a single-precision vector and a single-precision scalar value.
	void vDSP_vsmsma(const float *A, int AStride, const float *B, const float *C, int CStride, const float *D, float *E, int EStride, int N);
	// Adds a single-precision vector to the product of two single-precision vectors.
	void vDSP_vma(const float *A, int AStride, const float *B, int BStride, const float *C, int CStride, float *D, int DStride, int N);
	// Subtracts the product of two single-precision vectors from a second product of two single-precision vectors.
	void vDSP_vmmsb(const float *A, int AStride, const float *B, int BStride, const float *C, int CStride, const float *D, int DStride, float *E, int EStride, int N);
	// Multiplies the sum of two single-precision vectors by a second sum of two single-precision vectors. (A + B) * (C * D) -> E
	void vDSP_vaam(const float *A, int AStride, const float *B, int BStride, const float *C, int CStride, const float *D, int DStride, float *E, int EStride, int N);

	} // namespace CPPML
#endif


#endif