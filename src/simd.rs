//! SIMD playground for kNN-relevant kernels.
//!
//! Keep these small and isolated so you can experiment without touching core code paths.
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;
#[cfg(target_arch = "x86_64")]
type X86DistanceFn = unsafe fn(*const f32, *const f32, usize) -> f32;

/// Scalar squared Euclidean distance (reference).
pub fn squared_euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Auto-vectorization-friendly squared Euclidean distance.
///
/// Use this to see what LLVM does with a tight, predictable loop.
pub fn squared_euclidean_auto_vec(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    let mut i = 0usize;
    let len = a.len();
    while i < len {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }
    sum
}

/// Scalar dot product (reference).
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// NEON squared Euclidean distance (Apple Silicon).
///
/// Returns `None` if NEON is not available at runtime.
#[cfg(target_arch = "aarch64")]
pub fn squared_euclidean_neon(a: &[f32], b: &[f32]) -> Option<f32> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return None;
    }
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: guarded by runtime feature detection.
    Some(unsafe { squared_euclidean_neon_impl(a.as_ptr(), b.as_ptr(), a.len()) })
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn squared_euclidean_neon_impl(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vld1q_f32, vmulq_f32, vsubq_f32, vst1q_f32,
    };

    // acc = [0,0,0,0] accumulator vector
    let mut acc: float32x4_t = unsafe { vld1q_f32([0.0f32; 4].as_ptr()) };
    let mut i = 0usize;
    while i + 4 <= len {
        // Load 4 floats from a and b into SIMD registers.
        let va = unsafe { vld1q_f32(a.add(i)) };
        let vb = unsafe { vld1q_f32(b.add(i)) };
        // diff = va - vb (lane-wise)
        let diff = vsubq_f32(va, vb);
        // sq = diff * diff (lane-wise square)
        let sq = vmulq_f32(diff, diff);
        // acc += sq (lane-wise accumulation)
        acc = vaddq_f32(acc, sq);
        i += 4;
    }

    let mut tmp = [0.0f32; 4];
    // Store SIMD accumulator back to memory for horizontal sum.
    unsafe { vst1q_f32(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    while i < len {
        // Scalar tail for remaining elements.
        let diff = unsafe { *a.add(i) - *b.add(i) };
        sum += diff * diff;
        i += 1;
    }

    sum
}

/// SSE squared Euclidean distance (x86_64).
///
/// Returns `None` if SSE is not available at runtime.
#[cfg(target_arch = "x86_64")]
pub fn squared_euclidean_sse(a: &[f32], b: &[f32]) -> Option<f32> {
    if !has_x86_sse() {
        return None;
    }
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: guarded by runtime feature detection.
    Some(unsafe { squared_euclidean_sse_impl(a.as_ptr(), b.as_ptr(), a.len()) })
}

/// x86 runtime-dispatched squared Euclidean distance.
///
/// Prefers SSE over AVX on older Intel mobile CPUs where AVX can hurt
/// mixed scalar/SIMD workloads due to frequency and transition effects.
#[cfg(target_arch = "x86_64")]
pub fn squared_euclidean_x86_simd(a: &[f32], b: &[f32]) -> Option<f32> {
    debug_assert_eq!(a.len(), b.len());
    x86_distance_fn().map(|f| {
        // SAFETY: function pointer selected via runtime feature detection.
        unsafe { f(a.as_ptr(), b.as_ptr(), a.len()) }
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn squared_euclidean_sse_impl(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::{
        __m128, _mm_add_ps, _mm_hadd_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps,
        _mm_storeu_ps, _mm_sub_ps,
    };

    // acc = [0,0,0,0] accumulator vector
    let mut acc: __m128 = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= len {
        // Load 4 floats from a and b into SSE registers.
        let va = unsafe { _mm_loadu_ps(a.add(i)) };
        let vb = unsafe { _mm_loadu_ps(b.add(i)) };
        // diff = va - vb (lane-wise)
        let diff = _mm_sub_ps(va, vb);
        // sq = diff * diff (lane-wise square)
        let sq = _mm_mul_ps(diff, diff);
        // acc += sq (lane-wise accumulation)
        acc = _mm_add_ps(acc, sq);
        i += 4;
    }

    // Horizontal sum the 4 lanes into one scalar.
    let mut acc = unsafe { _mm_hadd_ps(acc, acc) };
    acc = unsafe { _mm_hadd_ps(acc, acc) };
    let mut tmp = [0.0f32; 4];
    unsafe { _mm_storeu_ps(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp[0];

    while i < len {
        // Scalar tail for remaining elements.
        let diff = unsafe { *a.add(i) - *b.add(i) };
        sum += diff * diff;
        i += 1;
    }

    sum
}

/// AVX squared Euclidean distance (x86_64).
///
/// Returns `None` if AVX is not available at runtime.
#[cfg(target_arch = "x86_64")]
pub fn squared_euclidean_avx(a: &[f32], b: &[f32]) -> Option<f32> {
    if !has_x86_avx() {
        return None;
    }
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: guarded by runtime feature detection.
    Some(unsafe { squared_euclidean_avx_impl(a.as_ptr(), b.as_ptr(), a.len()) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn squared_euclidean_avx_impl(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_hadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm256_zeroupper,
    };

    // acc = [0,0,0,0,0,0,0,0] accumulator vector
    let mut acc: __m256 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        // Load 8 floats from a and b into AVX registers.
        let va = unsafe { _mm256_loadu_ps(a.add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.add(i)) };
        // diff = va - vb (lane-wise)
        let diff = _mm256_sub_ps(va, vb);
        // sq = diff * diff (lane-wise square)
        let sq = _mm256_mul_ps(diff, diff);
        // acc += sq (lane-wise accumulation)
        acc = _mm256_add_ps(acc, sq);
        i += 8;
    }

    // Horizontal sum the 8 lanes into one scalar.
    let mut acc = _mm256_hadd_ps(acc, acc);
    acc = _mm256_hadd_ps(acc, acc);
    let mut tmp = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp[0] + tmp[4];

    while i < len {
        // Scalar tail for remaining elements.
        let diff = unsafe { *a.add(i) - *b.add(i) };
        sum += diff * diff;
        i += 1;
    }

    // Avoid AVX->SSE transition penalties in mixed scalar/SIMD callers.
    _mm256_zeroupper();

    sum
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_x86_sse() -> bool {
    static HAS_SSE: OnceLock<bool> = OnceLock::new();
    *HAS_SSE.get_or_init(|| is_x86_feature_detected!("sse"))
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_x86_avx() -> bool {
    static HAS_AVX: OnceLock<bool> = OnceLock::new();
    *HAS_AVX.get_or_init(|| is_x86_feature_detected!("avx"))
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_distance_fn() -> Option<X86DistanceFn> {
    static DIST_FN: OnceLock<Option<X86DistanceFn>> = OnceLock::new();
    *DIST_FN.get_or_init(|| {
        if has_x86_sse() {
            return Some(squared_euclidean_sse_impl);
        }
        if has_x86_avx() {
            return Some(squared_euclidean_avx_impl);
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::{dot_product_scalar, squared_euclidean_auto_vec, squared_euclidean_scalar};

    #[test]
    fn squared_euclidean_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 1.0, 0.0, 1.0, 2.0];
        let scalar = squared_euclidean_scalar(&a, &b);
        let auto = squared_euclidean_auto_vec(&a, &b);
        assert!((scalar - auto).abs() < 1e-6);
    }

    #[test]
    fn dot_product_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, -1.0, 2.0];
        let sum = dot_product_scalar(&a, &b);
        assert!((sum - 8.0).abs() < 1e-6);
    }
}
