#ifdef LLM_EXPT
extern void rmsnorm(
    const float* __restrict__ x, 
    const float* __restrict__ w, 
    float* __restrict__ y,
    int n
    );
#endif
