#ifdef LLM_EXPT
#include <immintrin.h>
#include <math.h>
#include "rmsnorm.h"


/**
 * Highly optimized RMSNorm implementation using AVX2 intrinsics.
 * * @param x  Input vector (aligned to 64-byte boundary)
 * @param w  Weight vector (aligned to 64-byte boundary)
 * @param n  Dimension/length of the vectors
 * @param y  Output vector (aligned to 64-byte boundary)
 */
void rmsnorm(
    const float* __restrict__ x, 
    const float* __restrict__ w, 
    float* __restrict__ y,
    int n
    ) 
{
    const float epsilon = 1e-5f;
    float sum_sq = 0.0f;
    int i = 0;

    // Initialize AVX2 accumulator for the sum of squares
    __m256 v_sum_sq = _mm256_setzero_ps();

    // Step 1: Vectorized Loop - Compute Sum of Squares (8 floats per iteration)
    for (; i <= n - 8; i += 8) {
        // Load input data using aligned load (as memory is 64-byte aligned)
        __m256 v_x = _mm256_load_ps(&x[i]);
        
        // Fused Multiply-Add: v_sum_sq = (v_x * v_x) + v_sum_sq
        v_sum_sq = _mm256_fmadd_ps(v_x, v_x, v_sum_sq);
    }

    // Horizontal add of the 8 elements in the AVX register to a single scalar
    float buffer[8];
    _mm256_store_ps(buffer, v_sum_sq);
    for (int j = 0; j < 8; ++j) {
        sum_sq += buffer[j];
    }

    // Handle any remaining elements if n is not a multiple of 8
    for (; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }

    // Step 2: Compute the scaling factor 's'
    // s = 1 / sqrt(epsilon + sum_sq / n) 
    // Note: Standard RMSNorm averages the sum of squares by 'n' before adding epsilon
    float mean_sq = sum_sq / (float)n;
    float s = 1.0f / sqrtf(epsilon + mean_sq);

    // Broadcast the scalar 's' to all 8 slots of an AVX register
    __m256 v_s = _mm256_set1_ps(s);

    // Step 3: Vectorized Loop - Compute y_i = w_i * x_i * s
    i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v_x = _mm256_load_ps(&x[i]);
        __m256 v_w = _mm256_load_ps(&w[i]);

        // Compute x_i * s
        __m256 v_xs = _mm256_mul_ps(v_x, v_s);
        
        // Compute w_i * (x_i * s)
        __m256 v_y = _mm256_mul_ps(v_w, v_xs);

        // Store the result back to y using aligned store
        _mm256_store_ps(&y[i], v_y);
    }

    // Handle remaining elements for the scaling phase
    for (; i < n; ++i) {
        y[i] = w[i] * x[i] * s;
    }
}

#endif
