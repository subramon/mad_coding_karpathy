#ifdef LLM_EXPT
extern int 
rope2(
      int kv_dim, 
      int head_size,
      int pos,
      float * restrict k // OUTPUT 
    );
#endif
