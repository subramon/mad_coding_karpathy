#ifdef LLM_EXPT
#include <math.h>
#include "rope2.h"
int 
rope2(
      int kv_dim, 
      int head_size,
      int pos,
      float * restrict k // OUTPUT 
    )
{
  int status = 0;

  for ( int i = 0; i < kv_dim; i+=2) {
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    float v0 = k[i];
    float v1 = k[i+1];
    k[i]   = v0 * fcr - v1 * fci;
    k[i+1] = v0 * fci + v1 * fcr;
  }
BYE:
  return status;
}
#endif
