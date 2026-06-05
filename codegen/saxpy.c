#include "saxpy.h"

#ifdef KARPATHY
void saxpy(
    float * const restrict x,
    const float * const restrict y,
    float a,
    int n
    )
{
  for (int i = 0; i < n; i++) {
    x[i] += a * y[i];
  }
}
#endif
#ifdef DEEPSEEK
#include <immintrin.h>
#include <stddef.h>

void saxpy(
    float * const restrict x,
    const float * const restrict y,
    float a,
    int n
    )
{
  // Broadcast scalar 'a' to all 8 lanes of an AVX2 vector
  __m256 a_vec = _mm256_set1_ps(a);

  // Process 8 floats per iteration (256 bits / 32 bits per float)
  const int vec_size = 8;
  int i = 0;

  // Main AVX2 loop: process full 8-float blocks
  for (; i <= n - vec_size; i += vec_size) {
    // Load 8 floats from x and y (unaligned allowed)
    __m256 x_vec = _mm256_loadu_ps(&x[i]);
    __m256 y_vec = _mm256_loadu_ps(&y[i]);

    // Compute a * y[i]
    __m256 a_y_vec = _mm256_mul_ps(a_vec, y_vec);

    // Compute x[i] + (a * y[i])
    __m256 result = _mm256_add_ps(x_vec, a_y_vec);

    // Store back to x (unaligned allowed)
    _mm256_storeu_ps(&x[i], result);
  }

  // Handle remaining elements (less than 8) with scalar code
  for (; i < n; ++i) {
    x[i] += a * y[i];
  }
}
#endif
