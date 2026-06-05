#include "vecmatmul.h"
#include <omp.h>
#ifdef KARPATHY
// neural net blocks; the dynamics of the Transformer
void vecmatmul(
    float * restrict xout, 
    const float * const restrict  x, 
    const float * const restrict w, 
    int n, 
    int d
    ) 
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}
#endif
#ifdef DEEPSEEK
#include <immintrin.h>
#include <stdlib.h>
#include <stddef.h>

void vecmatmul(
    float * restrict y, 
    const float * const restrict  x, 
    const float * const restrict w, 
    int n, 
    int m
    )
{

    // Process each row of the matrix
    for (int i = 0; i < m; ++i) {
        __m256 sum_vec = _mm256_setzero_ps();  // 8-wide vector accumulator

        int j = 0;

        // Process 8 elements at a time using AVX2
        for (; j <= n - 8; j += 8) {
            // Load x[j..j+7] (unaligned load)
            __m256 x_vec = _mm256_loadu_ps(&x[j]);

            // Load w[i][j..j+7] (unaligned load)
            __m256 w_vec = _mm256_loadu_ps(&w[i * n + j]);

            // Multiply and add to accumulator
            sum_vec = _mm256_fmadd_ps(w_vec, x_vec, sum_vec);
        }

        // Horizontal sum of the 8 values in sum_vec
        // Step 1: high-low addition
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 sum128 = _mm_add_ps(lo, hi);

        // Step 2: 4 -> 2
        __m128 shuf = _mm_movehdup_ps(sum128);
        __m128 sum64 = _mm_add_ps(sum128, shuf);

        // Step 3: 2 -> 1
        shuf = _mm_movehl_ps(shuf, sum64);
        float sum = _mm_cvtss_f32(_mm_add_ss(sum64, shuf));

        // Handle remaining elements (<8)
        for (; j < n; ++j) {
            sum += w[i * n + j] * x[j];
        }

        y[i] = sum;
    }
}
#endif
