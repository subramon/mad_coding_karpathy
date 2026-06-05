#include "vvincr.h"

#ifdef KARPATHY
void vvincr(
    float * restrict x,
    const float * const restrict y,
    int n
    )
{
  for (int i = 0; i < n; i++) {
    x[i] += y[i];
  }
}
#endif
#ifdef DEEPSEEK
#include <immintrin.h>
#include <stddef.h>

void vvincr(
    float * restrict x, 
    const float * const restrict y, 
    int n
    ) 
{
  // Process 8 floats at a time (256 bits / 32 bits per float)
  const int vec_size = 8;
  int i = 0;

  // Main AVX2 loop: handle as many full 8-float blocks as possible
  for (; i <= n - vec_size; i += vec_size) {
    // Load 8 floats from x and y (unaligned loads allowed)
    __m256 x_vec = _mm256_loadu_ps(&x[i]);
    __m256 y_vec = _mm256_loadu_ps(&y[i]);

    // Add vectors
    __m256 result = _mm256_add_ps(x_vec, y_vec);

    // Store back to x (unaligned store allowed)
    _mm256_storeu_ps(&x[i], result);
  }

  // Handle remaining elements (less than 8) with scalar code
  for (; i < n; ++i) {
    x[i] += y[i];
  }
}
#endif
