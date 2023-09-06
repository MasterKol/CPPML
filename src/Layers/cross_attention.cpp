#include "cross_attention.hpp"

#include <cmath>
#include <iostream>

#include "../LinearAlgebra.hpp"
#include "../random.hpp"

namespace CPPML {

void CrossAttention::init(int num_heads_, int qk_embed_size_, int v_embed_size_, int output_width, int Qwidth, int VKwidth, std::initializer_list<Layer*> Qs, std::initializer_list<Layer*> VKs){
	num_heads = num_heads_;
	qk_embed_size = qk_embed_size_;
	v_embed_size = v_embed_size_;
	output_shape.w(output_width);
	Q_shape.w(Qwidth);
	VK_shape.w(VKwidth);

	for(auto l = VKs.begin(); l != VKs.end(); l++){
		add_VK(*l);
	}

	for(auto l = Qs.begin(); l != Qs.end(); l++){
		add_Q(*l);
	}
}

void CrossAttention::add_VK(Layer* layer){
	add_input(layer);
	VK_layers.push_back(layer->get_output());
}

void CrossAttention::add_Q(Layer* layer){
	add_input(layer);
	Q_layers.push_back(layer->get_output());
}

void get_shape(std::vector<Layer*> Ls, Shape* shape){
	// try and use preset width if its > 0
	bool set_width = true;
	if(shape->w() <= 0){
		set_width = false;
		shape->w(Ls[0]->output_shape.w());
	}
	shape->h(0);

	// loop over layers and compute input shape
	for(Layer* l : Ls){
		if((!set_width && shape->w() != l->output_shape.w()) || (set_width && l->output_shape.size() % shape->w() != 0)){
			std::cerr << "Input widths do not match in cross attention\n";
			exit(-1);
		}
		shape->h(shape->h() + l->output_shape.size() / shape->w());
	}
}

bool CrossAttention::compile_(){
	if(inputs.size() != Q_layers.size() + VK_layers.size()){
		std::cerr << "CrossAttention: add_input was used to add input. Use add_Q or add_VK instead";
		exit(-1);
	}

	// clear inputs and add Q and VK inputs in correct input order
	inputs.clear();

	for(Layer* l : Q_layers){
		inputs.push_back(l);
	}

	for(Layer* l : VK_layers){
		inputs.push_back(l);
	}

	get_shape(Q_layers, &Q_shape);
	get_shape(VK_layers, &VK_shape);
	input_shape = Shape(Q_shape.size() + VK_shape.size());

	output_shape = Shape((output_shape.w() <= 0) ? Q_shape.w() : output_shape.w(), 
						Q_shape.h());

	intermediate_num = 0;
	// num heads * (Q_weight_size + K_weight_size + V_weight_size + Z_weight_size)
	num_params = num_heads * (Q_shape.w() * qk_embed_size + VK_shape.w() * qk_embed_size + VK_shape.w() * v_embed_size + v_embed_size * output_shape.w());

	return false;
}

void populate_mat(float** params, float** gradients, int w, int h, int d){
	float r = sqrtf(6.0f / (w + h));
	int len = w * h * d;
	Random::fillRand(*params, len, -r, r);

	*params += len;
	*gradients += len;
}

void CrossAttention::populate(float* params, float* gradients){
	q_mat = params;
	q_grads = gradients;
	populate_mat(&params, &gradients, qk_embed_size, Q_shape.w(), num_heads);

	v_mat = params;
	v_grads = gradients;
	populate_mat(&params, &gradients, v_embed_size, VK_shape.w(), num_heads);

	k_mat = params;
	k_grads = gradients;
	populate_mat(&params, &gradients, qk_embed_size, VK_shape.w(), num_heads);

	z_mat = params;
	z_grads = gradients;
	populate_mat(&params, &gradients, output_shape.w(), v_embed_size * num_heads, 1);
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

void CrossAttention::attention_head(float* Qin, float* VKin,
		float* qm, float* km, float* vm, float* zm,
		float* Q, float* K, float* KT, float* V, float* QKT,
		float* Z, float* O, const float norm_factor, bool calculateO){
	// compute K, K <- VKin * km
	vDSP_mmul(VKin, 1, km, 1, K, 1, VK_shape.h(), qk_embed_size, VK_shape.w());

	// transpose K
	vDSP_mtrans(K, 1, KT, 1, qk_embed_size, VK_shape.h());

	// compute Q, Q <- Qin * qm
	vDSP_mmul(Qin, 1, qm, 1, Q, 1, Q_shape.h(), qk_embed_size, Q_shape.w());

	// compute QKT, QKT <- Q * KT
	vDSP_mmul(Q, 1, KT, 1, QKT, 1, Q_shape.h(), VK_shape.h(), qk_embed_size);

	// multiply QKT by norm_factor
	vDSP_vsmul(QKT, 1, &norm_factor, QKT, 1, Q_shape.h() * VK_shape.h());

	// soft max QKT to get S
	for(int i = 0; i < Q_shape.h(); i++){
		softmax(QKT + i * VK_shape.h(), VK_shape.h());
	}

	// compute V, V <- VKin * vm
	vDSP_mmul(VKin, 1, vm, 1, V, 1, VK_shape.h(), v_embed_size, VK_shape.w());

	// compute Z, Z <- S * V
	vDSP_mmul(QKT, 1, V, 1, Z, 1, Q_shape.h(), v_embed_size, VK_shape.h());
	
	// conditionally calculate O
	if(calculateO){
		// O <- Z * zm
		vDSP_mmul(Z, 1, zm, 1, O, 1, Q_shape.h(), output_shape.w(), v_embed_size);
	}
}

void CrossAttention::compute(float* input, float* output, float* intermediate_buffer){
	// compute size of all matrices for later use
	const int Qw_size = qk_embed_size *  Q_shape.w();
	const int Vw_size =  v_embed_size * VK_shape.w();
	const int Kw_size = qk_embed_size * VK_shape.w();
	const int Zw_size = v_embed_size * output_shape.w();

	// zero output as it is added to not set
	memset(output, 0, output_shape.size() * sizeof(float));

	// buff needs to store max of
	// 2 * K, K + Q, V, and O
	float* const buff = new float[std::max({2 * VK_shape.h() * qk_embed_size, (VK_shape.h() + Q_shape.h()) * qk_embed_size, VK_shape.h() * v_embed_size, output_shape.size()})];

	// break out parts needed for processing
	float* const K = buff + qk_embed_size * VK_shape.h();
	float* const KT = buff;
	float* const Q = K;
	float* const V = buff;
	float* const O = buff;

	// Z and QKT get their own space as the space can't be used better AFAIK
	float* const Z = new float[Q_shape.h() * v_embed_size];
	float* const QKT = new float[Q_shape.h() * VK_shape.h()];

	// pointer to start of weight matrices, increments after each head
	float* q_mat_h = q_mat;
	float* v_mat_h = v_mat;
	float* k_mat_h = k_mat;
	float* z_mat_h = z_mat;

	// factor that QK^T is scaled by, the paper says to do
	// this but idk how necessary it is
	const float norm_factor = sqrt(1.0f / (float)qk_embed_size); // FIXME, use qk_embed or v_embed size?

	// loop over all heads
	for(int i = 0; i < num_heads; i++){
		attention_head(input, input + Q_shape.size(),
				q_mat_h, k_mat_h, v_mat_h, z_mat_h,
				Q, K, KT, V, QKT, Z, O, norm_factor, true);

		// out += O
		vDSP_vadd(O, 1, output, 1, output, 1, output_shape.size());

		// move to next layer
		q_mat_h += Qw_size;
		v_mat_h += Vw_size;
		k_mat_h += Kw_size;
		z_mat_h += Zw_size;
	}

	delete[] Z;
	delete[] QKT;
	delete[] buff;
}

void QVK_Derv(float* Pw, float* dPw, float* dP, float* inputT, float* dIn, float* buff, Shape InShape, int intSize){
	// dPw <- input^T * dP (this gives dPw)
	vDSP_mmul(inputT, 1, dP, 1, dPw, 1, InShape.w(), intSize, InShape.h());

	// dP <- dP^T
	vDSP_mtrans(dP, 1, buff, 1, intSize, InShape.h());
	memcpy(dP, buff, sizeof(float) * intSize * InShape.h());

	// buff (dIn^T) <- Pw * (dP^T)
	vDSP_mmul(Pw, 1, dP, 1, buff, 1, InShape.w(), InShape.h(), intSize);

	// dIn^T += buff
	vDSP_vadd(buff, 1, dIn, 1, dIn, 1, InShape.size());
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

void CrossAttention::get_change_grads(float* out_change, float* input_change,
				  float* input, float* output, float* intermediate){
	// compute size of all matrices for later use
	const int Qw_size = qk_embed_size *  Q_shape.w();
	const int Vw_size =  v_embed_size * VK_shape.w();
	const int Kw_size = qk_embed_size * VK_shape.w();
	const int Zw_size = v_embed_size * output_shape.w();

	// size of other elements
	const int Q_size = Q_shape.h() * qk_embed_size;
	const int V_size = VK_shape.h() * v_embed_size;
	const int K_size = VK_shape.h() * qk_embed_size;
	const int Z_size = Q_shape.h() * v_embed_size;
	const int S_size = Q_shape.h() * VK_shape.h();

	// zero input_change because backwards going gradients will
	// be added to it (it will never be assigned to)
	memset(input_change, 0, input_shape.size() * sizeof(float));

	// temp memory
	float* Q = new float[Q_size];
	float* K = new float[K_size];
	float* V = new float[V_size];
	float* QKT_sm = new float[S_size];
	float* Z = new float[Z_size];

	float* dZ = new float[Z_size];

	float* dP = new float[std::max({Q_size, K_size, V_size})];
	
	// pointer to start of weight matrices, increments after each head
	float* q_mat_h = q_mat;
	float* v_mat_h = v_mat;
	float* k_mat_h = k_mat;
	float* z_mat_h = z_mat;

	// get position of inputs
	float* Qin = input;
	float* VKin = input + Q_shape.size();
	
	// get position of backwards going gradients
	float* dQin = input_change;
	float* dVKin = input_change + Q_shape.size();

	// buffer for storing intermediate operations FIXME
	// buff needs to store
	// Q, K, V, Z, In, S
	const int buff_size = std::max({Q_size, K_size, V_size, Z_size, S_size, Q_shape.size(), VK_shape.size()});
	float* buff = new float[buff_size];

	// memory for storing temp storage gradients,
	// so they can be added later under mutex guard
	float* q_grd_t = new float[Qw_size];
	float* v_grd_t = new float[Vw_size];
	float* k_grd_t = new float[Kw_size];
	float* z_grd_t = new float[Zw_size];

	// generate transposed Qin
	float* QinT = new float[Q_shape.size()];
	vDSP_mtrans(Qin, 1, QinT, 1, Q_shape.w(), Q_shape.h());
	// generate transposed VKin
	float* VKinT = new float[VK_shape.size()];
	vDSP_mtrans(VKin, 1, VKinT, 1, VK_shape.w(), VK_shape.h());

	// factor that QK^T is scaled by, the paper says to do
	// this but idk how necessary it is
	const float inv_norm_factor = sqrtf((float)qk_embed_size);
	const float norm_factor = 1.0f / inv_norm_factor;

	for(int i = 0; i < num_heads; i++){
		attention_head(Qin, VKin, q_mat_h, k_mat_h, v_mat_h, z_mat_h, Q,
						K, buff, V, QKT_sm, Z, NULL, norm_factor, false);

		// buff <- Z^T
		vDSP_mtrans(Z, 1, buff, 1, v_embed_size, Q_shape.h());

		// t_grad(z_grad) <- buff(Z)^T * Out_change
		vDSP_mmul(buff, 1, out_change, 1, z_grd_t, 1, v_embed_size, output_shape.w(), Q_shape.h());

		// buff <- z_weights^T
		vDSP_mtrans(z_mat_h, 1, buff, 1, output_shape.w(), v_embed_size);

		// dZ = out_change * z_weights^T (out_change * buff)
		vDSP_mmul(out_change, 1, buff, 1, dZ, 1, Q_shape.h(), v_embed_size, output_shape.w());

		// buff <- QKT_sm^T
		vDSP_mtrans(QKT_sm, 1, buff, 1, VK_shape.h(), Q_shape.h());

		// dP <- QKT_sm^T * dZ   (dV)
		vDSP_mmul(buff, 1, dZ, 1, dP, 1, VK_shape.h(), v_embed_size, Q_shape.h());

		// process dP   (dV)
		QVK_Derv(v_mat_h, v_grd_t, dP, VKinT, dVKin, buff, VK_shape, v_embed_size);

		// transpose V into dP temporarily
		vDSP_mtrans(V, 1, dP, 1, v_embed_size, VK_shape.h());
		// buff <- dZ * buff(V)^T (compute dS)
		vDSP_mmul(dZ, 1, dP, 1, buff, 1, Q_shape.h(), VK_shape.h(), v_embed_size);

		// compute dQK^T from dS and S and write it to S
		// dS is in buff atm
		float* dS = buff;
		float* S  = QKT_sm;
		for(int j = 0; j < Q_shape.h(); j++){
			d_softmax(S, dS, VK_shape.h());

			// dQKT *= inv_norm_scale (undo scaling)
			// do it here because maybe its better for the cache?
			vDSP_vsmul(S, 1, &inv_norm_factor, S, 1, VK_shape.h());

			dS += VK_shape.h();
			S  += VK_shape.h();
		}

		// dont' need to transpose K because it is already in the correct form

		// dP (dQ) = dQKT * K
		vDSP_mmul(QKT_sm, 1, K, 1, dP, 1, Q_shape.h(), qk_embed_size, VK_shape.h());

		// process dQ
		QVK_Derv(q_mat_h, q_grd_t, dP, QinT, dQin, buff, Q_shape, qk_embed_size);

		// dQKT <- dQKT^T
		vDSP_mtrans(QKT_sm, 1, buff, 1, VK_shape.h(), Q_shape.h());
		// dP (dK) = dQKT^T * Q
		vDSP_mmul(buff, 1, Q, 1, dP, 1, VK_shape.h(), qk_embed_size, Q_shape.h());

		// process dK
		QVK_Derv(k_mat_h, k_grd_t, dP, VKinT, dVKin, buff, VK_shape, qk_embed_size);

		q_mat_h += Qw_size;
		v_mat_h += Vw_size;
		k_mat_h += Kw_size;
		z_mat_h += Zw_size;

		// add temporary storage of gradients to main gradients, claim
		// lock to preserve thread safety
		gradient_mutex.lock();

		vDSP_vadd(q_grd_t, 1, q_grads + Qw_size * i, 1, q_grads + Qw_size * i, 1, Qw_size);
		vDSP_vadd(v_grd_t, 1, v_grads + Vw_size * i, 1, v_grads + Vw_size * i, 1, Vw_size);
		vDSP_vadd(k_grd_t, 1, k_grads + Kw_size * i, 1, k_grads + Kw_size * i, 1, Kw_size);
		vDSP_vadd(z_grd_t, 1, z_grads + Zw_size * i, 1, z_grads + Zw_size * i, 1, Zw_size);

		gradient_mutex.unlock();
	}

	// un-transpose dQ and dVK
	// dQ <- (dQ^T)^T
	vDSP_mtrans(dQin, 1, buff, 1, Q_shape.h(), Q_shape.w());
	memcpy(dQin, buff, Q_shape.size() * sizeof(float));

	// dVK <- (dVK^T)^T
	vDSP_mtrans(dVKin, 1, buff, 1, VK_shape.h(), VK_shape.w());
	memcpy(dVKin, buff, VK_shape.size() * sizeof(float));

	// free all allocated memory
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
	delete[] QinT;
	delete[] VKinT;
}

} // namespace CPPML