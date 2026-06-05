#include "softmax.h"

// Exactly one of these should be defined

#ifdef DEEPSEEK
#include <math.h>
void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}
#endif
#ifdef KARPATHY
#include <immintrin.h>
#include <math.h>
#include <stddef.h>


// AVX2 exp approximation (accurate enough for softmax)
static inline __m256 exp256_ps(__m256 x) {
    // exp(x) = 2^(x * log2(e))
    // Using polynomial approximation for 2^x in [-1, 1] range
    __m256 log2e = _mm256_set1_ps(1.442695041f);
    __m256 scaled = _mm256_mul_ps(x, log2e);

    // Split into integer and fractional parts
    __m256 int_part = _mm256_round_ps(scaled, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    __m256 frac_part = _mm256_sub_ps(scaled, int_part);

    // Polynomial approximation for 2^x on [-0.5, 0.5]
    // f(x) = 1 + x * (ln2 + x * (ln2^2/2 + x * (ln2^3/6 + x * ln2^4/24)))
    const __m256 c1 = _mm256_set1_ps(0.69314718056f);  // ln2
    const __m256 c2 = _mm256_set1_ps(0.240226507f);   // ln2^2/2
    const __m256 c3 = _mm256_set1_ps(0.055504108f);   // ln2^3/6
    const __m256 c4 = _mm256_set1_ps(0.009618129f);   // ln2^4/24

    __m256 result = _mm256_fmadd_ps(c4, frac_part, c3);
    result = _mm256_fmadd_ps(result, frac_part, c2);
    result = _mm256_fmadd_ps(result, frac_part, c1);
    result = _mm256_fmadd_ps(result, frac_part, _mm256_set1_ps(1.0f));

    // Scale by 2^int_part
    __m256i int_part_int = _mm256_cvtps_epi32(int_part);
    __m256i exp_int = _mm256_slli_epi32(_mm256_add_epi32(int_part_int, _mm256_set1_epi32(127)), 23);
    __m256 scale = _mm256_castsi256_ps(exp_int);

    return _mm256_mul_ps(result, scale);
}
void softmax(float* x, int n) {
    if (n <= 0) return;

    // Step 1: Find max value
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    int i = 0;

    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        __m256 vals = _mm256_loadu_ps(x + i);  // unaligned load
        max_vec = _mm256_max_ps(max_vec, vals);
    }

    // Reduce vector to scalar max
    float max_arr[8];
    _mm256_storeu_ps(max_arr, max_vec);
    float max_val = max_arr[0];
    for (int j = 1; j < 8; j++) {
        if (max_arr[j] > max_val) max_val = max_arr[j];
    }

    // Handle remaining elements
    for (; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Step 2: Compute sum of exp(x_i - max)
    i = 0;
    __m256 sum_vec = _mm256_setzero_ps();

    for (; i + 7 < n; i += 8) {
        __m256 vals = _mm256_loadu_ps(x + i);
        __m256 shifted = _mm256_sub_ps(vals, _mm256_set1_ps(max_val));
        __m256 exp_vals = exp256_ps(shifted);  // exp for 8 floats
        sum_vec = _mm256_add_ps(sum_vec, exp_vals);
        _mm256_storeu_ps(x + i, exp_vals);  // store exp values temporarily
    }

    // Reduce sum vector
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_vec);
    float total_sum = sum_arr[0];
    for (int j = 1; j < 8; j++) total_sum += sum_arr[j];

    // Handle remaining elements
    for (; i < n; i++) {
        float exp_val = expf(x[i] - max_val);
        total_sum += exp_val;
        x[i] = exp_val;
    }

    // Step 3: Normalize by sum
    float inv_sum = 1.0f / total_sum;
    i = 0;

    for (; i + 7 < n; i += 8) {
        __m256 vals = _mm256_loadu_ps(x + i);
        __m256 normalized = _mm256_mul_ps(vals, _mm256_set1_ps(inv_sum));
        _mm256_storeu_ps(x + i, normalized);
    }

    for (; i < n; i++) {
        x[i] *= inv_sum;
    }
}
#endif
