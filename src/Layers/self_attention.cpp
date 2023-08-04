#include "self_attention.hpp"
#include "../helper.hpp"

#include <cmath>
#include "../LinearAlgebra.hpp"
#include <assert.h>

bool SelfAttention::compile_(){
	bool set_width = true;

	if(input_shape.w <= 0){
		set_width = false;
		input_shape.w = inputs[0]->output_shape.w;
	}
	input_shape.h = 0;

	for(Layer* l : inputs){
		if((!set_width && input_shape.w != l->output_shape.w) || (set_width && l->output_shape.size % input_shape.w != 0)){
			throw std::runtime_error("Input widths don't match");
		}
		input_shape.h += l->output_shape.size / input_shape.w;
	}
	input_shape.fix_size();

	output_shape = Shape((output_shape.w <= 0) ? input_shape.w : output_shape.w, 
						input_shape.h);

	intermediate_num = 0;
	num_params = num_heads * internal_size * (input_shape.w * 3 + output_shape.w);

	return false;
}

// handles setting param pointers and initializing each of the
// parameter matrices, this is broken out for simplicity
inline void mat_setup(float** p_dst, float** g_dst, float* p_src, float* g_src, int mat_size, int num){
	*p_dst = p_src + mat_size * num;
	*g_dst = g_src + mat_size * num;
}

void SelfAttention::populate(float* params, float* gradients){
	const int mat_size = input_shape.w * internal_size * num_heads;

	mat_setup(&q_mat, &q_grads, params, gradients, mat_size, 0);
	mat_setup(&v_mat, &v_grads, params, gradients, mat_size, 1);
	mat_setup(&k_mat, &k_grads, params, gradients, mat_size, 2);

	mat_setup(&z_mat, &z_grads, params, gradients, mat_size, 3);

	// this num is chosen to keep variance at 1
	const float qvk_range = sqrtf(6.0f / (input_shape.w + internal_size));
	//const float qvk_sdv = sqrtf(0.5f / (input_shape.w));
	for(int i = 0; i < mat_size * 3; i++){
		params[i] = randF(-qvk_range, qvk_range);
	}

	// this num is chosen to keep variance at 1
	const float z_range = sqrt(6.0f / (internal_size * num_heads + output_shape.w));
	//const float z_sdv = sqrtf(0.5f / (num_heads * internal_size));
	for(int i = 0; i < internal_size * num_heads * output_shape.w; i++){
		z_mat[i] = randF(-z_range, z_range);
	}
}

// performs softmax on the given vector of length N
static void softmax(float* v, int N){
	// this slightly odd implementation of softmax gives more
	// numerical stability than direct application of the formula

	float max;
	// find the maximum value in v
	vDSP_maxv(v, 1, &max, N);

	// subtract max off of v
	max = -max;
	vDSP_vsadd(v, 1, &max, v, 1, N);

	// exp v
	vvexpf(v, v, &N);

	// sum v
	float total;
	vDSP_sve(v, 1, &total, N);

	// divide v by total
	vDSP_vsdiv(v, 1, &total, v, 1, N);
}

inline void SelfAttention::attention_head(float* input,
		float* qm, float* km, float* vm, float* zm,
		float* Q, float* K, float* V, float* QKT,
		float* Z, float* O, const float norm_factor, bool calculateO){
	// create K (but write it to Q temporarily)
	vDSP_mmul(input, 1, km, 1, Q, 1, input_shape.h, internal_size, input_shape.w);
	// transpose K (drawn from Q)
	vDSP_mtrans(Q, 1, K, 1, internal_size, input_shape.h);
	
	// create Q
	vDSP_mmul(input, 1, qm, 1, Q, 1, input_shape.h, internal_size, input_shape.w);

	// QKT <- Q * K^T
	vDSP_mmul(Q, 1, K, 1, QKT, 1, input_shape.h, input_shape.h, internal_size);

	// divide QKT by norm_factor
	vDSP_vsmul(QKT, 1, &norm_factor, QKT, 1, input_shape.h * input_shape.h);

	// softmax each row of K
	float* QK_row = QKT;
	for(int i = 0; i < input_shape.h; i++){
		softmax(QK_row, input_shape.h);
		QK_row += input_shape.h;
	}

	// make V
	vDSP_mmul(input, 1, vm, 1, V, 1, input_shape.h, internal_size, input_shape.w);

	// Z <- QKT * V
	vDSP_mmul(QKT, 1, V, 1, Z, 1, input_shape.h, internal_size, input_shape.h);

	// O is not needed for gradients so make this optional
	if(calculateO){
		// O <- Z * Z_mat
		vDSP_mmul(Z, 1, zm, 1, O, 1, input_shape.h, output_shape.w, internal_size);
	}
}

void SelfAttention::compute(float* input, float* output, float* intermediate_buffer){
	// size of Q, V, and K
	const int QVK_size = internal_size * input_shape.h;
	// input height squared
	const int ih_sq = input_shape.h * input_shape.h;
	// size of a single slice of a q/v/k weight matrix
	const int qvk_weight_size = internal_size * input_shape.w;
	// size of a single slide of the z weight matrix
	const int z_weight_size = internal_size * output_shape.w;

	// zero output as it is added to not set
	memset(output, 0, output_shape.size * sizeof(float));

	// buffer for intermediate memory
	const int buf_size = QVK_size + std::max(QVK_size + ih_sq, output_shape.size);
	float* const buff = new float[buf_size];

	// some of these memory regions overlap but they are
	// guaranteed to not overwrite each other during use
	float* const Q = buff;
	float* const K = buff + QVK_size;
	float* const V = buff + QVK_size;

	float* const QKT = buff + 2 * QVK_size;
	
	float* const Z = buff;
	float* const O = buff + QVK_size;

	// pointer to start of weight matrices, increments after each head
	float* q_mat_h = q_mat;
	float* v_mat_h = v_mat;
	float* k_mat_h = k_mat;
	float* z_mat_h = z_mat;

	// factor that QK^T is scaled by, the paper says to do
	// this but idk how necessary it is
	const float norm_factor = sqrt(1.0f / (float)internal_size);

	// loop over all heads
	for(int i = 0; i < num_heads; i++){
		attention_head(input, q_mat_h, k_mat_h, v_mat_h, z_mat_h, Q, K, V, 
						QKT, Z, O, norm_factor, true);

		// out += O
		vDSP_vadd(O, 1, output, 1, output, 1, output_shape.size);

		// move to next layer
		q_mat_h += qvk_weight_size;
		v_mat_h += qvk_weight_size;
		k_mat_h += qvk_weight_size;
		z_mat_h += z_weight_size;
	}

	delete[] buff;
}

inline void SelfAttention::QVK_Derv(float* Pw, float* dPw, float* dP, float* inputT, float* dIn, float* buff, bool dP_transposed){
	// dPw <- input^T * dP (this gives dPw)
	vDSP_mmul(inputT, 1, dP, 1, dPw, 1, input_shape.w, internal_size, input_shape.h);

	if(!dP_transposed){
		// dP <- dP^T
		vDSP_mtrans(dP, 1, buff, 1, internal_size, input_shape.h);
		memcpy(dP, buff, sizeof(float) * internal_size * input_shape.h);
	}

	// buff (dIn^T) <- Pw * (dP^T)
	vDSP_mmul(Pw, 1, dP, 1, buff, 1, input_shape.w, input_shape.h, internal_size);

	// dIn^T += buff
	vDSP_vadd(buff, 1, dIn, 1, dIn, 1, input_shape.size);
}

static void d_softmax(float* val, float* grad, int N){
	float dt;
	// dt = dS . S
	vDSP_dotpr(grad, 1, val, 1, &dt, N);
	
	// dS -= dt
	dt = -dt;
	vDSP_vsadd(grad, 1, &dt, grad, 1, N);
	
	// dQKT = S .* dS (write to S)
	vDSP_vmul(val, 1, grad, 1, val, 1, N);
}

void SelfAttention::get_change_grads(float* out_change, float* input_change,
				  float* input, float* output, float* intermediate){
	// size of Q, V, and K
	const int QVK_size = internal_size * input_shape.h;
	// input height squared
	const int ih_sq = input_shape.h * input_shape.h;
	// size of a single slice of a q/v/k weight matrix
	const int qvk_weight_size = internal_size * input_shape.w;
	// size of a single slide of the z weight matrix
	const int z_weight_size = internal_size * output_shape.w;

	// zero input_change because backwards going gradients will
	// be added to it (it will never be assigned to)
	memset(input_change, 0, input_shape.size * sizeof(float));

	// temp memory
	float* const Q = new float[QVK_size];
	float* const K = new float[QVK_size];
	float* const V = new float[QVK_size];
	float* const QKT_sm = new float[ih_sq];
	float* const Z = new float[QVK_size];

	float* const dZ = new float[QVK_size];
	float* const dP = new float[QVK_size];

	// buffer for storing intermediate operations
	const int buff_size = std::max(std::max(qvk_weight_size, z_weight_size), std::max(ih_sq, input_shape.size));
	float* const buff = new float[buff_size];

	// pointer to start of weight matrices, increments after each head
	float* q_mat_h = q_mat;
	float* v_mat_h = v_mat;
	float* k_mat_h = k_mat;
	float* z_mat_h = z_mat;

	// memory for storing temp storage gradients,
	// so they can be added later under mutex guard
	float* const q_grd_t = new float[qvk_weight_size];
	float* const v_grd_t = new float[qvk_weight_size];
	float* const k_grd_t = new float[qvk_weight_size];
	float* const z_grd_t = new float[  z_weight_size];

	// generate transposed input
	float* const inT = new float[input_shape.size];
	vDSP_mtrans(input, 1, inT, 1, input_shape.w, input_shape.h);

	// factor that QK^T is scaled by, the paper says to do
	// this but idk how necessary it is
	const float inv_norm_factor = sqrtf((float)internal_size);
	const float norm_factor = 1.0f / inv_norm_factor;

	for(int i = 0; i < num_heads; i++){
		attention_head(input, q_mat_h, k_mat_h, v_mat_h, z_mat_h, Q, K, V, 
						QKT_sm, Z, NULL, norm_factor, false);

		// buff <- Z^T
		vDSP_mtrans(Z, 1, buff, 1, internal_size, input_shape.h);

		// z_grd_t <- buff(Z)^T * Out_change
		vDSP_mmul(buff, 1, out_change, 1, z_grd_t, 1, internal_size, output_shape.w, input_shape.h);

		// buff <- z_weights^T
		vDSP_mtrans(z_mat_h, 1, buff, 1, output_shape.w, internal_size);
		
		// dZ = out_change * z_weights^T (out_change * buff)
		vDSP_mmul(out_change, 1, buff, 1, dZ, 1, input_shape.h, internal_size, output_shape.w);

		// buff <- QKT_sm^T
		vDSP_mtrans(QKT_sm, 1, buff, 1, input_shape.h, input_shape.h);

		// dP <- QKT_sm^T * dZ   (dV)
		vDSP_mmul(buff, 1, dZ, 1, dP, 1, input_shape.h, internal_size, input_shape.h);

		// process dP   (dV)
		QVK_Derv(v_mat_h, v_grd_t, dP, inT, input_change, buff);

		// transpose V into dP temporarily
		vDSP_mtrans(V, 1, dP, 1, internal_size, input_shape.h);
		// buff <- dZ * buff(V)^T (compute dS)
		vDSP_mmul(dZ, 1, dP, 1, buff, 1, input_shape.h, input_shape.h, internal_size);
		// V is now free for other use

		// compute dQK^T from dS and S and write it to S
		// dS is in buff atm
		float* dS = buff;
		float* S  = QKT_sm;
		for(int j = 0; j < input_shape.h; j++){
			d_softmax(S, dS, input_shape.h);

			// dQKT *= inv_norm_scale (undo scaling)
			// do it here because maybe its better for the cache?
			vDSP_vsmul(S, 1, &inv_norm_factor, S, 1, input_shape.h);

			dS += input_shape.h;
			S  += input_shape.h;
		}

		// Transpose K to take it from K^T back to K
		vDSP_mtrans(K, 1, V, 1, input_shape.h, internal_size);

		// dP (dQ) = dQKT * K  (V is used instead of K because I previously stored K^T in V)
		vDSP_mmul(QKT_sm, 1, V, 1, dP, 1, input_shape.h, internal_size, input_shape.h);

		// process dQ
		QVK_Derv(q_mat_h, q_grd_t, dP, inT, input_change, buff);
		
		// buff <- dQKT^T
		vDSP_mtrans(QKT_sm, 1, buff, 1, input_shape.h, input_shape.h);
		// dP (dK) = dQKT^T * Q
		vDSP_mmul(buff, 1, Q, 1, dP, 1, input_shape.h, internal_size, input_shape.h);

		// process dK
		QVK_Derv(k_mat_h, k_grd_t, dP, inT, input_change, buff);

		// move to next layer
		q_mat_h += qvk_weight_size;
		v_mat_h += qvk_weight_size;
		k_mat_h += qvk_weight_size;
		z_mat_h += z_weight_size;

		// add temporary storage of gradients to main gradients, claim
		// lock to preserve thread safety
		const int qvk_offset = qvk_weight_size * i;
		const int z_offset = z_weight_size * i;
		gradient_mutex.lock();

		vDSP_vadd(q_grd_t, 1, q_grads + qvk_offset, 1, q_grads + qvk_offset, 1, qvk_weight_size);
		vDSP_vadd(v_grd_t, 1, v_grads + qvk_offset, 1, v_grads + qvk_offset, 1, qvk_weight_size);
		vDSP_vadd(k_grd_t, 1, k_grads + qvk_offset, 1, k_grads + qvk_offset, 1, qvk_weight_size);
		vDSP_vadd(z_grd_t, 1, z_grads +   z_offset, 1, z_grads +   z_offset, 1,   z_weight_size);

		gradient_mutex.unlock();
	}

	// dIn <- (dIn^T)^T
	vDSP_mtrans(input_change, 1, buff, 1, input_shape.h, input_shape.w);
	memcpy(input_change, buff, input_shape.size * sizeof(float));

	// free memory
	delete[] Q;
	delete[] K;
	delete[] V;
	delete[] QKT_sm;
	delete[] Z;
	delete[] dZ;
	delete[] dP;
	delete[] buff;
	delete[] q_grd_t;
	delete[] v_grd_t;
	delete[] k_grd_t;
	delete[] z_grd_t;
	delete[] inT;
}