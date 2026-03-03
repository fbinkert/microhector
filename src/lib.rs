#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

/// Represents a single search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityResult {
    pub id: u64,
    pub distance: f32,
}

#[derive(Default)]
pub struct VectorDatabase<const DIM: usize>(Vec<(u64, [f32; DIM])>);

#[derive(Default)]
pub struct VectorDatabaseSoA<const DIM: usize> {
    ids: Vec<u64>,
    data: Vec<f32>,
}

impl<const DIM: usize> VectorDatabase<DIM> {
    /// Insert a vector with an ID.
    pub fn insert(&mut self, id: u64, vector: [f32; DIM]) {
        self.0.push((id, vector));
    }

    /// Search for the top K nearest neighbors (naive full sort).
    #[must_use]
    pub fn search(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        self.search_naive(query, k)
    }

    /// Search for the top K nearest neighbors (naive full sort).
    #[must_use]
    pub fn search_naive(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        let mut result: Vec<SimilarityResult> = self
            .0
            .iter()
            .map(|(id, vector)| SimilarityResult {
                id: *id,
                // Squared Euclidean distance
                distance: squared_euclidean::<DIM>(query, vector),
            })
            .collect();

        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result.truncate(k);
        result
    }

    /// Search for the top K nearest neighbors (naive full sort, SIMD when available).
    #[must_use]
    pub fn search_naive_simd(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        let mut result: Vec<SimilarityResult> = self
            .0
            .iter()
            .map(|(id, vector)| SimilarityResult {
                id: *id,
                distance: squared_euclidean_simd::<DIM>(query, vector),
            })
            .collect();

        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result.truncate(k);
        result
    }

    /// Search for the top K nearest neighbors (naive full sort, portable SIMD).
    #[must_use]
    pub fn search_naive_portable_simd(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        let mut result: Vec<SimilarityResult> = self
            .0
            .iter()
            .map(|(id, vector)| SimilarityResult {
                id: *id,
                distance: squared_euclidean_portable_simd::<DIM>(query, vector),
            })
            .collect();

        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result.truncate(k);
        result
    }

    /// Search for the top K nearest neighbors using a bounded max-heap.
    #[must_use]
    pub fn search_topk_heap(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        if k == 0 || self.0.is_empty() {
            return Vec::new();
        }

        let mut heap: std::collections::BinaryHeap<HeapItem> =
            std::collections::BinaryHeap::with_capacity(k);

        for (id, vector) in &self.0 {
            let distance = squared_euclidean::<DIM>(query, vector);
            if heap.len() < k {
                heap.push(HeapItem { id: *id, distance });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(HeapItem { id: *id, distance });
                }
            }
        }

        let mut result: Vec<SimilarityResult> = heap
            .into_iter()
            .map(|item| SimilarityResult {
                id: item.id,
                distance: item.distance,
            })
            .collect();
        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result
    }

    /// Search for the top K nearest neighbors using a bounded max-heap (SIMD when available).
    #[must_use]
    pub fn search_topk_heap_simd(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        if k == 0 || self.0.is_empty() {
            return Vec::new();
        }

        let mut heap: std::collections::BinaryHeap<HeapItem> =
            std::collections::BinaryHeap::with_capacity(k);

        for (id, vector) in &self.0 {
            let distance = squared_euclidean_simd::<DIM>(query, vector);
            if heap.len() < k {
                heap.push(HeapItem { id: *id, distance });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(HeapItem { id: *id, distance });
                }
            }
        }

        let mut result: Vec<SimilarityResult> = heap
            .into_iter()
            .map(|item| SimilarityResult {
                id: item.id,
                distance: item.distance,
            })
            .collect();
        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result
    }

    /// Search for the top K nearest neighbors using a bounded max-heap (portable SIMD).
    #[must_use]
    pub fn search_topk_heap_portable_simd(
        &self,
        query: &[f32; DIM],
        k: usize,
    ) -> Vec<SimilarityResult> {
        if k == 0 || self.0.is_empty() {
            return Vec::new();
        }

        let mut heap: std::collections::BinaryHeap<HeapItem> =
            std::collections::BinaryHeap::with_capacity(k);

        for (id, vector) in &self.0 {
            let distance = squared_euclidean_portable_simd::<DIM>(query, vector);
            if heap.len() < k {
                heap.push(HeapItem { id: *id, distance });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(HeapItem { id: *id, distance });
                }
            }
        }

        let mut result: Vec<SimilarityResult> = heap
            .into_iter()
            .map(|item| SimilarityResult {
                id: item.id,
                distance: item.distance,
            })
            .collect();
        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result
    }
}

impl<const DIM: usize> VectorDatabaseSoA<DIM> {
    /// Insert a vector with an ID.
    pub fn insert(&mut self, id: u64, vector: [f32; DIM]) {
        self.ids.push(id);
        self.data.extend_from_slice(&vector);
    }

    /// Search for the top K nearest neighbors (naive full sort).
    #[must_use]
    pub fn search_naive(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        let mut result: Vec<SimilarityResult> = self
            .ids
            .iter()
            .enumerate()
            .map(|(idx, id)| {
                let start = idx * DIM;
                let end = start + DIM;
                let vector = &self.data[start..end];
                SimilarityResult {
                    id: *id,
                    distance: squared_euclidean_slice(query, vector),
                }
            })
            .collect();

        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result.truncate(k);
        result
    }

    /// Search for the top K nearest neighbors (naive full sort, SIMD when available).
    #[must_use]
    pub fn search_naive_simd(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        let mut result: Vec<SimilarityResult> = self
            .ids
            .iter()
            .enumerate()
            .map(|(idx, id)| {
                let start = idx * DIM;
                let end = start + DIM;
                let vector = &self.data[start..end];
                SimilarityResult {
                    id: *id,
                    distance: squared_euclidean_simd_slice(query, vector),
                }
            })
            .collect();

        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result.truncate(k);
        result
    }

    /// Search for the top K nearest neighbors (naive full sort, portable SIMD).
    #[must_use]
    pub fn search_naive_portable_simd(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        let mut result: Vec<SimilarityResult> = self
            .ids
            .iter()
            .enumerate()
            .map(|(idx, id)| {
                let start = idx * DIM;
                let end = start + DIM;
                let vector = &self.data[start..end];
                SimilarityResult {
                    id: *id,
                    distance: squared_euclidean_portable_simd_slice(query, vector),
                }
            })
            .collect();

        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result.truncate(k);
        result
    }

    /// Search for the top K nearest neighbors using a bounded max-heap.
    #[must_use]
    pub fn search_topk_heap(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        if k == 0 || self.ids.is_empty() {
            return Vec::new();
        }

        let mut heap: std::collections::BinaryHeap<HeapItem> =
            std::collections::BinaryHeap::with_capacity(k);

        for (idx, id) in self.ids.iter().enumerate() {
            let start = idx * DIM;
            let end = start + DIM;
            let vector = &self.data[start..end];
            let distance = squared_euclidean_slice(query, vector);
            if heap.len() < k {
                heap.push(HeapItem { id: *id, distance });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(HeapItem { id: *id, distance });
                }
            }
        }

        let mut result: Vec<SimilarityResult> = heap
            .into_iter()
            .map(|item| SimilarityResult {
                id: item.id,
                distance: item.distance,
            })
            .collect();
        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result
    }

    /// Search for the top K nearest neighbors using a bounded max-heap (SIMD when available).
    #[must_use]
    pub fn search_topk_heap_simd(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        if k == 0 || self.ids.is_empty() {
            return Vec::new();
        }

        let mut heap: std::collections::BinaryHeap<HeapItem> =
            std::collections::BinaryHeap::with_capacity(k);

        for (idx, id) in self.ids.iter().enumerate() {
            let start = idx * DIM;
            let end = start + DIM;
            let vector = &self.data[start..end];
            let distance = squared_euclidean_simd_slice(query, vector);
            if heap.len() < k {
                heap.push(HeapItem { id: *id, distance });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(HeapItem { id: *id, distance });
                }
            }
        }

        let mut result: Vec<SimilarityResult> = heap
            .into_iter()
            .map(|item| SimilarityResult {
                id: item.id,
                distance: item.distance,
            })
            .collect();
        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result
    }

    /// Search for the top K nearest neighbors using a bounded max-heap (portable SIMD).
    #[must_use]
    pub fn search_topk_heap_portable_simd(
        &self,
        query: &[f32; DIM],
        k: usize,
    ) -> Vec<SimilarityResult> {
        if k == 0 || self.ids.is_empty() {
            return Vec::new();
        }

        let mut heap: std::collections::BinaryHeap<HeapItem> =
            std::collections::BinaryHeap::with_capacity(k);

        for (idx, id) in self.ids.iter().enumerate() {
            let start = idx * DIM;
            let end = start + DIM;
            let vector = &self.data[start..end];
            let distance = squared_euclidean_portable_simd_slice(query, vector);
            if heap.len() < k {
                heap.push(HeapItem { id: *id, distance });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(HeapItem { id: *id, distance });
                }
            }
        }

        let mut result: Vec<SimilarityResult> = heap
            .into_iter()
            .map(|item| SimilarityResult {
                id: item.id,
                distance: item.distance,
            })
            .collect();
        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result
    }
}

#[inline]
fn squared_euclidean<const DIM: usize>(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    squared_euclidean_ptr(a.as_ptr(), b.as_ptr(), DIM)
}

#[inline]
fn squared_euclidean_simd<const DIM: usize>(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    squared_euclidean_simd_ptr(a.as_ptr(), b.as_ptr(), DIM)
}

#[inline]
fn squared_euclidean_portable_simd<const DIM: usize>(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    squared_euclidean_portable_simd_slice(a.as_slice(), b.as_slice())
}

#[inline]
fn squared_euclidean_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    squared_euclidean_ptr(a.as_ptr(), b.as_ptr(), a.len())
}

#[inline]
fn squared_euclidean_simd_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    squared_euclidean_simd_ptr(a.as_ptr(), b.as_ptr(), a.len())
}

#[inline]
fn squared_euclidean_portable_simd_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "portable-simd")]
    {
        use std::simd::num::SimdFloat;
        use std::simd::Simd;

        const LANES: usize = 8;
        type Vf32 = Simd<f32, LANES>;

        let mut acc = Vf32::splat(0.0);
        let mut i = 0usize;
        let len = a.len();

        while i + LANES <= len {
            let va = Vf32::from_slice(&a[i..i + LANES]);
            let vb = Vf32::from_slice(&b[i..i + LANES]);
            let diff = va - vb;
            acc += diff * diff;
            i += LANES;
        }

        let mut sum = acc.reduce_sum();
        while i < len {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }

        return sum;
    }

    #[cfg(not(feature = "portable-simd"))]
    {
        squared_euclidean_slice(a, b)
    }
}

#[inline]
fn squared_euclidean_ptr(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    while i < len {
        // SAFETY: caller guarantees a/b valid for len.
        let diff = unsafe { *a.add(i) - *b.add(i) };
        sum += diff * diff;
        i += 1;
    }
    sum
}

#[inline]
fn squared_euclidean_simd_ptr(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { squared_euclidean_avx(a, b, len) };
        }
        if is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { squared_euclidean_sse(a, b, len) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { squared_euclidean_neon(a, b, len) };
        }
    }

    squared_euclidean_ptr(a, b, len)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn squared_euclidean_avx(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_hadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let mut acc: __m256 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        acc = _mm256_add_ps(acc, sq);
        i += 8;
    }

    // Horizontal sum acc.
    let mut acc = _mm256_hadd_ps(acc, acc);
    acc = _mm256_hadd_ps(acc, acc);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[4];

    while i < len {
        let diff = *a.add(i) - *b.add(i);
        sum += diff * diff;
        i += 1;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn squared_euclidean_sse(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::{
        __m128, _mm_add_ps, _mm_hadd_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps,
        _mm_storeu_ps, _mm_sub_ps,
    };

    let mut acc: __m128 = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.add(i));
        let vb = _mm_loadu_ps(b.add(i));
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        acc = _mm_add_ps(acc, sq);
        i += 4;
    }

    let mut acc = _mm_hadd_ps(acc, acc);
    acc = _mm_hadd_ps(acc, acc);
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0];

    while i < len {
        let diff = *a.add(i) - *b.add(i);
        sum += diff * diff;
        i += 1;
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn squared_euclidean_neon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vld1q_f32, vmulq_f32, vsubq_f32, vst1q_f32,
    };

    let mut acc: float32x4_t = unsafe { vld1q_f32([0.0f32; 4].as_ptr()) };
    let mut i = 0usize;
    while i + 4 <= len {
        let va = unsafe { vld1q_f32(a.add(i)) };
        let vb = unsafe { vld1q_f32(b.add(i)) };
        let diff = vsubq_f32(va, vb);
        let sq = vmulq_f32(diff, diff);
        acc = vaddq_f32(acc, sq);
        i += 4;
    }

    let mut tmp = [0.0f32; 4];
    unsafe { vst1q_f32(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    while i < len {
        let diff = unsafe { *a.add(i) - *b.add(i) };
        sum += diff * diff;
        i += 1;
    }

    sum
}

#[derive(Debug, Copy, Clone)]
struct HeapItem {
    id: u64,
    distance: f32,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.total_cmp(&other.distance)
    }
}
