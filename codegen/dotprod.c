
#include "dotprod.h"


#ifdef KARPATHY
float dotprod(
    const float* const restrict x, 
    const float* const restrict y, 
    int n
    ) 
{
  float rslt = 0;
  for ( int i = 0; i < n; i++ ) { 
    rslt += x[i]* y[i];
  }
  return rslt;
}
#endif
#ifdef DEEPSEEK
#include <immintrin.h>

float dotprod(
    const float* const restrict x, 
    const float* const restrict y, 
    int n
    ) 
{
    __m256 sum_vec = _mm256_setzero_ps();  // 8-lane vector accumulator
    int i = 0;

    // Process 8 elements at a time (unaligned loads)
    for (; i <= n - 8; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x[i]);  // unaligned load
        __m256 y_vec = _mm256_loadu_ps(&y[i]);  // unaligned load
        sum_vec = _mm256_fmadd_ps(x_vec, y_vec, sum_vec);
    }

    // Horizontal sum of 8 lanes in sum_vec
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(lo, hi);   // 8 -> 4

    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sum64 = _mm_add_ps(sum128, shuf); // 4 -> 2

    shuf = _mm_movehl_ps(shuf, sum64);
    float result = _mm_cvtss_f32(_mm_add_ss(sum64, shuf)); // 2 -> 1

    // Handle remaining elements (<8)
    for (; i < n; ++i) {
        result += x[i] * y[i];
    }

    return result;
}
#endif
