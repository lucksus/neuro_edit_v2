//! CUDA C source compiled at run time with NVRTC (cudarc). Mirrors `cpu.rs`
//! step-for-step; differences are only float-atomic summation order in the
//! delay ring (documented in NOTES).
//!
//! The neuron model is a switch point: `//@MODEL_STEP@` inside
//! `neuron_kernel` is replaced with one dispatch case per model in the
//! net's model table (`kernel_source`); each model file owns its step
//! text (`NeuronModel::cuda_step`).

use crate::model::NeuronModel;

/// The placeholder inside `KERNELS_SRC` where per-model step text lands.
const STEP_MARK: &str = "//@MODEL_STEP@";

/// Assemble the kernel source for a net's model table. The cases are
/// emitted in table order — a one-model net gets the step body with no
/// dispatch at all, so today's speed stays. `int_weights` selects the
/// fixed-point delay ring (bitwise-reproducible accumulation).
pub fn kernel_source(models: &[&dyn NeuronModel], int_weights: bool) -> String {
    let cases = if models.len() == 1 {
        format!(
            "I += nzk * ({});\n{}",
            models[0].cuda_noise_gain(),
            models[0].cuda_step()
        )
    } else {
        let mut s = String::new();
        for (k, m) in models.iter().enumerate() {
            if k > 0 {
                s.push_str(" else ");
            }
            s.push_str(&format!(
                "if(mi == {k}){{\nI += nzk * ({});\n{}\n}}",
                m.cuda_noise_gain(),
                m.cuda_step()
            ));
        }
        s
    };
    let src = KERNELS_SRC.replace(STEP_MARK, &cases);
    if int_weights {
        format!("#define INT_RING 1\n{src}")
    } else {
        src
    }
}

pub const KERNELS_SRC: &str = r#"
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

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
// xstate/params/model carry the generic model blocks: neuron i's state
// row is xstate[i*X..], param row params[i*P..], table index model[i].
// xt_ptr/xt_idx/xsyn/xdecay: per-synapse current traces for synapses with
// their own tau (C custom synapses, target-major rows like a CSR).
extern "C" __global__ void neuron_kernel(
    int N, u64 t, int D, float dt,
    float pkick, float kick,
    float* xstate, const float* params, const u8* model, int P, int X,
    float* s, float* trace,
    rt_t* ring,
    const float* ext,
    const float* stim, long long stim_t0, int n_inputs,
    const u32* stim_ptr, const u32* stim_input, const float* stim_w,
    float decay_s, float decay_t,
    u64 seed,
    int* spk, int* cnt_cur,
    u64* spk_row,
    u64* pn_total,
    const u32* vwatch, int nwatch, float* xrec, int vrec_d,
    const int* gmap, int n_grids, u32* gls, const u32* goff,
    const u32* xt_ptr, const u32* xt_idx, float* xsyn, const float* xdecay)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= N) return;
  int slot = (int)(t % (u64)D);
  float arr = rt_get(ring[(u64)slot * N + i]);
  ring[(u64)slot * N + i] = rt_put(0.f);
  float sv = s[i] + arr;
  float I = sv + ext[i];
  // per-synapse traces into this neuron: sum for input, decay in the
  // same pass (decay-after-use, same as the merged s path)
  for(u32 j = xt_ptr[i]; j < xt_ptr[i+1]; ++j){
    u32 k = xt_idx[j];
    I += xsyn[k];
    xsyn[k] *= xdecay[k];
  }

  // stimulus
  long long rel = (long long)t - stim_t0;
  if(rel >= 0){
    const float* stim_row = stim + (u64)rel * n_inputs;
    for(u32 j = stim_ptr[i]; j < stim_ptr[i+1]; ++j){
      I += stim_w[j] * stim_row[stim_input[j]];
    }
  }
  // background Poisson kicks: nzk is the raw one-step pulse (kick/dt
  // input units); each model case scales it so a kick of `kick` is
  // `noise_kick_mv` mV of depolarisation (AdEx: *C, izh: *1)
  float nzk = (pkick > 0.f
      && (float)noise_hash((u32)i, t, seed) * 2.3283064e-10f < pkick)
    ? kick : 0.f;

  // the model step sees p[k], x[k], I, dt and sets `spiked`; x/p are
  // this thread's rows of the flat blocks
  int mi = model[i];
  float* x = xstate + (u64)i * X;
  const float* p = params + (u64)i * P;
  bool spiked = false;
  //@MODEL_STEP@
  // per-step state record for watched neurons (v(t) + v–u phase plots):
  // one ring row holds all X state slots of the watched neuron
  for(int w = 0; w < nwatch; ++w){
    if((u32)i == vwatch[w]){
      u64 slot = (u64)(t % (u64)vrec_d) * (u64)nwatch * (u64)X
               + (u64)w * (u64)X;
      for(int k = 0; k < X; ++k) xrec[slot + k] = x[k];
    }
  }

  if(spiked){
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
// source) and scatters weights into the delay ring — custom-tau synapses
// go to the per-synapse arrival ring instead of the merged per-neuron one.
// Also: thread 0 resets the NEXT step's counter slot and accumulates the
// total spike count — that is why there is no per-step host readback.
extern "C" __global__ void prop_kernel(
    int N, u64 t, int D,
    const int* spk, const int* cnt_cur, int* cnt_next,
    u64* spikes_total,
    const u32* src_ptr, const u32* tgt, const float* w, const u16* delay,
    rt_t* ring,
    const u32* xsyn_map, int C, rt_t* xring)
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
      u32 xid = xsyn_map[j];
      if(xid != 0xFFFFFFFFu){
        atomicAdd(&xring[(u64)slot * (u64)C + (u64)xid], rt_put(w[j]));
      }else{
        atomicAdd(&ring[(u64)slot * N + tgt[j]], rt_put(w[j]));
      }
    }
  }
}

// One thread per custom synapse: fold this step's arrivals into the
// trace (x += w) and clear the slot — runs before neuron_kernel.
extern "C" __global__ void xsyn_arrive_kernel(
    int C, int slot, rt_t* xring, float* xsyn)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j >= C) return;
  xsyn[j] += rt_get(xring[(u64)slot * (u64)C + (u64)j]);
  xring[(u64)slot * (u64)C + (u64)j] = rt_put(0.f);
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
