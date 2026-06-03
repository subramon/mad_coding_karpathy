#ifdef LLM_EXPT
#include <math.h>
#include "rope1.h"
int 
rope1(
      int dim,
      int head_size,
      int pos,
      float * restrict q // OUTPUT 
    )
{
  int status = 0;

  for ( int i = 0; i < dim; i+=2) {
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float v0 = q[i];
    float v1 = q[i+1];
    q[i]   = v0 * fcr - v1 * fci;
    q[i+1] = v0 * fci + v1 * fcr;
  }
BYE:
  return status;
}
#endif
