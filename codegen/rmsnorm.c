#include "rmsnorm.h"
#ifdef KARPATHY
#include <math.h>
void rmsnorm(
    float* restrict y, 
    const float* restrict x, 
    const float* restrict w, 
    int n
    ) 
{
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < n; j++) {
        ss += x[j] * x[j];
    }
    ss /= n;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < n; j++) {
        y[j] = w[j] * (ss * x[j]);
    }
}
#endif
#ifdef GEMINI
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
void rmsnorm(
    float* restrict y, 
    const float* restrict x, 
    const float* restrict w, 
    int n
    ) 
{
    // 1. Define epsilon as requested
    const float epsilon = 1e-5f; // 10^-5

    float sum_sq = 0.0f;
    int i = 0;

    // Accumulate sum of squares using AVX2 if possible
    // Processing 8 floats at a time
    if (n >= 8) {
        __m256 v_sum_sq = _mm256_setzero_ps();

        for (; i <= n - 8; i += 8) {
            // Load unaligned data from x
            __m256 v_x = _mm256_loadu_ps(&x[i]);

            // Multiply x_i * x_i and add to running total vector
            v_sum_sq = _mm256_fmadd_ps(v_x, v_x, v_sum_sq);
        }

        // Horizontal add of the 8 elements in v_sum_sq
        // Extract high and low 128-bit lanes
        __m128 v_low = _mm256_castps256_ps128(v_sum_sq);
        __m128 v_high = _mm256_extractf128_ps(v_sum_sq, 1);
        __m128 v_sum128 = _mm_add_ps(v_low, v_high);

        // Final horizontal sum reductions within the 128-bit vector
        v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);
        v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);

        _mm_store_ss(&sum_sq, v_sum128);
    }

    // Scalar cleanup loop for remaining elements (or if n < 8)
    for (; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }

    // 2. Compute scale factor: s = 1 / sqrt(epsilon + (sum_sq / n))
    float mean_sq = sum_sq / (float)n;
    float s = 1.0f / sqrtf(epsilon + mean_sq);

    // 3. Compute y_i = w_i * x_i * s
    // Reset index to process vector elements again
    i = 0;
    if (n >= 8) {
        // Broadcast the scalar 's' across all 8 lanes
        __m256 v_s = _mm256_set1_ps(s);

        for (; i <= n - 8; i += 8) {
            // Load unaligned data from x and w
            __m256 v_x = _mm256_loadu_ps(&x[i]);
            __m256 v_w = _mm256_loadu_ps(&w[i]);

            // Multiply x_i * s
            __m256 v_xs = _mm256_mul_ps(v_x, v_s);

            // Multiply (x_i * s) * w_i
            __m256 v_y = _mm256_mul_ps(v_xs, v_w);

            // Store unaligned data back to y
            _mm256_storeu_ps(&y[i], v_y);
        }
    }

    // Scalar cleanup loop for remaining elements (or if n < 8)
    for (; i < n; ++i) {
        y[i] = w[i] * x[i] * s;
    }
}
#endif
#ifdef DEEPSEEK
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
void rmsnorm(
    float* restrict y, 
    const float* restrict x, 
    const float* restrict w, 
    int n
    ) 
{
    const float eps = 1e-5f;

    // Step 1: Compute sum of squares: sum(x[i] * x[i])
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;

    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        // Load 8 floats from x (unaligned)
        __m256 x_vec = _mm256_loadu_ps(x + i);

        // Square each element: x_vec * x_vec
        __m256 sq_vec = _mm256_mul_ps(x_vec, x_vec);

        // Accumulate squares
        sum_vec = _mm256_add_ps(sum_vec, sq_vec);
    }

    // Horizontal sum of the 8 values in sum_vec
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    // Handle remaining elements (less than 8)
    for (; i < n; i++) {
        sum += x[i] * x[i];
    }

    // Step 2: Compute s = 1 / sqrt(eps + sum / n)
    // Note: The formula in the doc appears to have an error.
    // Standard RMSNorm uses: s = 1 / sqrt(eps + mean(x^2))
    // where mean(x^2) = sum(x_i^2) / n
    float mean_sq = sum / (float)n;
    float s = 1.0f / sqrtf(eps + mean_sq);

    // Step 3: Compute y[i] = w[i] * x[i] * s
    i = 0;

    // Broadcast s to all 8 lanes
    __m256 s_vec = _mm256_set1_ps(s);

    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        // Load x and w (unaligned)
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 w_vec = _mm256_loadu_ps(w + i);

        // Compute x * s
        __m256 x_scaled = _mm256_mul_ps(x_vec, s_vec);

        // Compute w * (x * s)
        __m256 y_vec = _mm256_mul_ps(w_vec, x_scaled);

        // Store result (unaligned)
        _mm256_storeu_ps(y + i, y_vec);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        y[i] = w[i] * x[i] * s;
    }
}
#endif
