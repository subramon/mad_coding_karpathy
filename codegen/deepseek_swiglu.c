#include "swiglu.h"

#ifdef KARPATHY
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
#endif
#ifdef DEEPSEEK // TODO PO Actually Gemini
        #include <immintrin.h>

// Fast approximate exp(x) for x in [-10, 10] (sigmoid range)
static inline __m256 _mm256_exp_ps_approx(__m256 x) {
    // Clamp to avoid overflow
    x = _mm256_min_ps(x, _mm256_set1_ps(10.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-10.0f));

    // exp(x) = 2^(x / ln(2))
    __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    __m256 ln2_inv = _mm256_set1_ps(1.4426950408889634f);

    // n = round(x / ln(2))
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, ln2_inv), _MM_FROUND_TO_NEAREST_INT);

    // Remainder: r = x - n * ln(2)
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2));

    // Polynomial approximation for 2^r on [0, 1]
    // p(r) = 1 + (ln2)*r + (ln2)^2/2 * r^2 + (ln2)^3/6 * r^3 + ...
    __m256 ln2_2 = _mm256_set1_ps(0.240226507f);   // ln2^2/2
    __m256 ln2_3 = _mm256_set1_ps(0.055504108f);   // ln2^3/6
    __m256 ln2_4 = _mm256_set1_ps(0.009618129f);   // ln2^4/24

    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 r3 = _mm256_mul_ps(r2, r);
    __m256 r4 = _mm256_mul_ps(r3, r);

    __m256 p = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(ln2, r));
    p = _mm256_add_ps(p, _mm256_mul_ps(ln2_2, r2));
    p = _mm256_add_ps(p, _mm256_mul_ps(ln2_3, r3));
    p = _mm256_add_ps(p, _mm256_mul_ps(ln2_4, r4));

    // Compute 2^n = ldexp(1.0f, n)
    __m256 exp_val = _mm256_scalef_ps(p, n);  // AVX2: p * 2^n

    return exp_val;
}

void swiglu(float*  restrict x, const float* restrict y, int n) {
    __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;

    for (; i <= n - 8; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        __m256 y_vec = _mm256_loadu_ps(&y[i]);

        // Sigmoid(x) = 1 / (1 + exp(-x))
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x_vec);
        __m256 exp_neg_x = _mm256_exp_ps_approx(neg_x);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));

        // x = x * y * sigmoid(x)
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(x_vec, y_vec), sigmoid);
        _mm256_storeu_ps(&x[i], result);
    }

    for (; i < n; ++i) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[i]));
        x[i] = x[i] * y[i] * sigmoid_x;
    }
}
#endif
