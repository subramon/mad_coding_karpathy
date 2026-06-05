#include "swiglu.h"

#ifdef KARPATHY // TODO PO 
#include <math.h>
void
swiglu(
    float * restrict x,
    const float * const restrict y,
    int n
    )
{
  for (int i = 0; i < n; i++) {
    float val = x[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-val)));
    // elementwise multiply with w3(x)
    val *= y[i];
    x[i] = val;
  }
}
#endif
#ifdef GEMINI // Gets strange results 
#include <immintrin.h>
#include <math.h>

void swiglu(float* restrict x, const float* restrict y, int n) {
    // Process 8 elements at a time using AVX2
    int i = 0;

    // Constants for the exponential approximation: e^x = 2^(x * log2(e))
    __m256 log2e  = _mm256_set1_ps(1.4426950408889634f);
    __m256 ln2_hi = _mm256_set1_ps(-6.93145751953125e-1f);
    __m256 ln2_lo = _mm256_set1_ps(-1.4286068207448243e-7f);

    // Polynomial coefficients for e^t on t in [-0.5, 0.5]
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(0.5f);
    __m256 c3 = _mm256_set1_ps(1.666666666e-1f);
    __m256 c4 = _mm256_set1_ps(4.166666791e-2f);
    __m256 c5 = _mm256_set1_ps(8.333333763e-3f);

    __m256 one = _mm256_set1_ps(1.0f);

    for (; i <= n - 8; i += 8) {
        // Load x and y (unaligned)
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_loadu_ps(&y[i]);

        // --- Fast AVX2 Exp Evaluation: exp(vx) ---
        // r = round(vx * log2e)
        __m256 fx = _mm256_mul_ps(vx, log2e);
        __m256 r  = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

        // t = vx - r * ln2 (using split constants for precision)
        __m256 t = _mm256_fmadd_ps(r, ln2_hi, vx);
        t = _mm256_fmadd_ps(r, ln2_lo, t);

        // Evaluate polynomial: c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*c5))))
        __m256 p = c5;
        p = _mm256_fmadd_ps(p, t, c4);
        p = _mm256_fmadd_ps(p, t, c3);
        p = _mm256_fmadd_ps(p, t, c2);
        p = _mm256_fmadd_ps(p, t, c1);
        p = _mm256_fmadd_ps(p, t, c0);

        // Calculate 2^r by shifting the rounded float into the exponent bits
        // conversion to integer is safe assuming moderate ranges of x
        __m256i imm_g = _mm256_cvtps_epi32(r);
        imm_g = _mm256_add_epi32(imm_g, _mm256_set1_epi32(127));
        imm_g = _mm256_slli_epi32(imm_g, 23);
        __m256 pow2r = _mm256_castsi256_ps(imm_g);

        // exp(vx) = p * 2^r
        __m256 vexp = _mm256_mul_ps(p, pow2r);

        // --- SwiGLU Calculation ---
        // denom = 1.0 + exp(vx)
        __m256 denom = _mm256_add_ps(one, vexp);

        // sig = 1.0 / (1.0 + exp(vx))
        __m256 sig = _mm256_div_ps(one, denom);

        // x_i = x_i * y_i * sig
        __m256 vprod = _mm256_mul_ps(vx, vy);
        __m256 vres  = _mm256_mul_ps(vprod, sig);

        // Store back to x (unaligned)
        _mm256_storeu_ps(&x[i], vres);
    }

    // Clean up remaining tail elements sequentially
    for (; i < n; ++i) {
        x[i] = x[i] * y[i] * (1.0f / (1.0f + expf(x[i])));
    }
}
#endif
#ifdef DEEPSEEK 
#include <math.h>
#include <immintrin.h>

// Fast approximate exp(x) using AVX2 only (no AVX512)
static inline __m256 _mm256_exp_ps_avx2(__m256 x) {
    // Clamp to range [-10, 10] to avoid overflow
    __m256 min_val = _mm256_set1_ps(-10.0f);
    __m256 max_val = _mm256_set1_ps(10.0f);
    x = _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);

    // exp(x) = 2^(x / ln(2))
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 ln2_inv = _mm256_set1_ps(1.4426950408889634f);

    // n = floor(x / ln(2) + 0.5)
    __m256 n = _mm256_mul_ps(x, ln2_inv);
    n = _mm256_round_ps(n, _MM_FROUND_TO_NEAREST_INT);

    // r = x - n * ln(2)
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2));

    // Polynomial: exp(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666667f);
    const __m256 c4 = _mm256_set1_ps(0.04166667f);
    const __m256 c5 = _mm256_set1_ps(0.00833333f);

    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 r3 = _mm256_mul_ps(r2, r);
    __m256 r4 = _mm256_mul_ps(r3, r);
    __m256 r5 = _mm256_mul_ps(r4, r);

    __m256 exp_r = _mm256_add_ps(c1, r);
    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c2, r2));
    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c3, r3));
    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c4, r4));
    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c5, r5));

    // Compute 2^n = exp2(n) using bit manipulation
    __m256i n_int = _mm256_cvtps_epi32(n);
    n_int = _mm256_add_epi32(n_int, _mm256_set1_epi32(127));
    n_int = _mm256_slli_epi32(n_int, 23);
    __m256 exp2_n = _mm256_castsi256_ps(n_int);

    // exp(x) = exp(r) * 2^n
    return _mm256_mul_ps(exp_r, exp2_n);
}

void swiglu(float* x, const float* y, int n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    int i = 0;

    // Process 8 elements at a time
    for (; i <= n - 8; i += 8) {
        // Load x[i..i+7] and y[i..i+7] (unaligned)
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        __m256 y_vec = _mm256_loadu_ps(&y[i]);

        // Compute sigmoid(x) = 1 / (1 + exp(-x))
        __m256 neg_x = _mm256_sub_ps(zero, x_vec);
        __m256 exp_neg_x = _mm256_exp_ps_avx2(neg_x);
        __m256 denom = _mm256_add_ps(one, exp_neg_x);
        __m256 sigmoid = _mm256_div_ps(one, denom);

        // Compute result: x * y * sigmoid(x)
        __m256 result = _mm256_mul_ps(x_vec, y_vec);
        result = _mm256_mul_ps(result, sigmoid);

        // Store result back to x[i..i+7] (unaligned)
        _mm256_storeu_ps(&x[i], result);
    }

    // Handle remaining elements (n % 8)
    for (; i < n; ++i) {
        float sigmoid_x = 1.0f / (1.0f + expf(x[i]));
        x[i] = x[i] * y[i] * sigmoid_x;
    }
}
#endif
