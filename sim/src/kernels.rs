//! CUDA C source compiled at run time with NVRTC (cudarc). Mirrors `cpu.rs`
//! step-for-step; differences are only float-atomic summation order in the
//! delay ring (documented in NOTES).
//!
//! The neuron model is a switch point: `model == 0` is Izhikevich; further
//! models add a `case` and a params interpretation.

pub const KERNELS_SRC: &str = r#"
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;

// Delay-ring element type. With INT_RING the ring is 32-bit fixed point
// (WSCALE 4096): int atomicAdd is order-independent, so runs are bitwise
// reproducible — the float ring is not.
#ifdef INT_RING
  #define WSCALE 4096.f
  typedef int rt_t;
  __device__ __forceinline__ float rt_get(rt_t x){ return (float)x / WSCALE; }
  __device__ __forceinline__ rt_t rt_put(float w){ return (int)lrintf(w * WSCALE); }
#else
  typedef float rt_t;
  __device__ __forceinline__ float rt_get(rt_t x){ return x; }
  __device__ __forceinline__ rt_t rt_put(float w){ return w; }
#endif

// Same mixing as state::noise_hash on the CPU (splitmix64).
__device__ __forceinline__ u32 noise_hash(u32 i, u64 t, u64 seed){
  u64 h = (u64)i * 0x9E3779B97F4A7C15ull
        + t * 0xBF58476D1CE4E5B9ull
        + seed * 0x94D049BB133111EBull;
  h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 27; h *= 0x94D049BB133111EBull;
  h ^= h >> 31;
  return (u32)(h >> 32);
}

// One neuron per thread. t is the absolute step index.
extern "C" __global__ void neuron_kernel(
    int N, u64 t, int D, float dt,
    float pkick, float kick,
    float* v, float* u, float* s, float* trace,
    const float* a, const float* b, const float* c, const float* d,
    rt_t* ring,
    const float* ext,
    const float* stim, long long stim_t0, int n_inputs,
    const u32* stim_ptr, const u32* stim_input, const float* stim_w,
    float decay_s, float decay_t,
    u64 seed,
    int model,
    int* spk, int* cnt_cur,
    u64* spk_row,
    u64* pn_total,
    const u32* vwatch, int nwatch, float* vrec, int vrec_d,
    const int* gmap, int n_grids, u32* gls, const u32* goff)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= N) return;
  int slot = (int)(t % (u64)D);
  float arr = rt_get(ring[(u64)slot * N + i]);
  ring[(u64)slot * N + i] = rt_put(0.f);
  float sv = s[i] + arr;
  float I = sv + ext[i];

  // stimulus
  long long rel = (long long)t - stim_t0;
  if(rel >= 0){
    const float* stim_row = stim + (u64)rel * n_inputs;
    for(u32 j = stim_ptr[i]; j < stim_ptr[i+1]; ++j){
      I += stim_w[j] * stim_row[stim_input[j]];
    }
  }
  // background Poisson kicks
  if(pkick > 0.f && (float)noise_hash((u32)i, t, seed) * 2.3283064e-10f < pkick)
    I += kick;

  float vv = v[i], uu = u[i];
  if(model == 0){
    // Izhikevich (identical order to the CPU reference and the 2011 original)
    if(vv >= 30.f){ vv = c[i]; uu += d[i]; }
    vv += dt * (0.04f*vv*vv + 5.f*vv + 140.f - uu + I);
    uu += dt * a[i] * (b[i]*vv - uu);
  }
  v[i] = vv; u[i] = uu;
  // per-step v record for watched neurons (editor v(t) plot)
  for(int w = 0; w < nwatch; ++w){
    if((u32)i == vwatch[w]) vrec[(u64)(t % (u64)vrec_d) * (u64)nwatch + (u64)w] = vv;
  }

  if(vv >= 30.f){
    int k = atomicAdd(cnt_cur, 1);
    spk[k] = i;
    trace[i] = 1.f;
    atomicAdd(pn_total + i, (u64)1);
    if(spk_row != 0){
      atomicOr(spk_row + (i >> 6), (u64)1 << (i & 63));
    }
    // mark the texels this neuron belongs to (editor plane view): each
    // grid group's last-spike-step array is materialized on readback.
    for(int g = 0; g < n_grids; ++g){
      int tex = gmap[(u64)g * (u64)N + (u64)i];
      if(tex >= 0) gls[(u64)goff[g] + (u64)tex] = (u32)t;
    }
  }else{
    trace[i] *= decay_t;
  }
  s[i] = sv * decay_s;
}

// One thread per possible spike; iterates the spike's out-degree (CSR by
// source) and scatters weights into the delay ring.
// Also: thread 0 resets the NEXT step's counter slot and accumulates the
// total spike count — that is why there is no per-step host readback.
extern "C" __global__ void prop_kernel(
    int N, u64 t, int D,
    const int* spk, const int* cnt_cur, int* cnt_next,
    u64* spikes_total,
    const u32* src_ptr, const u32* tgt, const float* w, const u16* delay,
    rt_t* ring)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid == 0){
    cnt_next[0] = 0;
    atomicAdd(spikes_total, (u64)cnt_cur[0]);
  }
  int lane = tid & 31;
  int warp = tid >> 5;
  int nwarps = (gridDim.x * blockDim.x) >> 5;
  int cnt = cnt_cur[0];
  for(int s = warp; s < cnt; s += nwarps){
    int i = spk[s];
    u32 lo = src_ptr[i], hi = src_ptr[i + 1];
    for(u32 j = lo + lane; j < hi; j += 32){
      int slot = (int)((t + (u64)delay[j]) % (u64)D);
      atomicAdd(&ring[(u64)slot * N + tgt[j]], rt_put(w[j]));
    }
  }
}

// Materialize one grid group's spike-trace buffer for readback:
// out[texel] = exp(-(t - last_spike_step) * k), 0 for never-spiked.
extern "C" __global__ void grid_trace_kernel(
    const u32* ls, int M, u64 t, float k, float* out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= M) return;
  u32 l = ls[i];
  out[i] = (l == 0xFFFFFFFFu) ? 0.f
        : expf(-(float)(t - (u64)l) * k);
}
"#;
