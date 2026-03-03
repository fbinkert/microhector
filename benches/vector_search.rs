use std::hint::black_box;

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use microhector::{VectorDatabase, VectorDatabaseSoA, simd};
use rand::RngExt;

const DIM: usize = 128;

fn bench_brute_force_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(3));

    let mut rng = rand::rng();

    for num_records in [10_000u64, 100_000u64] {
        let mut db = VectorDatabase::<DIM>::default();
        let mut db_soa = VectorDatabaseSoA::<DIM>::default();

        // Populate the database
        for i in 0..num_records {
            let vector: [f32; DIM] = std::array::from_fn(|_| rng.random());
            db.insert(i, vector);
            db_soa.insert(i, vector);
        }

        // Create a random query vector
        let query: [f32; DIM] = std::array::from_fn(|_| rng.random());
        let top_k = 10;

        group.bench_with_input(
            BenchmarkId::new("aos_naive_full_sort", num_records),
            &num_records,
            |b, _| {
            b.iter(|| {
                let results = db.search_naive(black_box(&query), black_box(top_k));
                black_box(results);
            });
        },
        );

        group.bench_with_input(
            BenchmarkId::new("aos_naive_full_sort_simd", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results = db.search_naive_simd(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("aos_naive_portable_simd", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results =
                        db.search_naive_portable_simd(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("aos_topk_heap", num_records), &num_records, |b, _| {
            b.iter(|| {
                let results = db.search_topk_heap(black_box(&query), black_box(top_k));
                black_box(results);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("aos_topk_heap_simd", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results = db.search_topk_heap_simd(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("aos_topk_heap_portable_simd", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results =
                        db.search_topk_heap_portable_simd(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("soa_naive_full_sort", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results = db_soa.search_naive(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("soa_naive_full_sort_simd", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results = db_soa.search_naive_simd(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("soa_naive_portable_simd", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results =
                        db_soa.search_naive_portable_simd(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("soa_topk_heap", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results = db_soa.search_topk_heap(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("soa_topk_heap_simd", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results = db_soa.search_topk_heap_simd(black_box(&query), black_box(top_k));
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("soa_topk_heap_portable_simd", num_records),
            &num_records,
            |b, _| {
                b.iter(|| {
                    let results = db_soa.search_topk_heap_portable_simd(
                        black_box(&query),
                        black_box(top_k),
                    );
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_simd_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_kernels");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    const LEN: usize = 1024;
    let mut rng = rand::rng();
    let a: Vec<f32> = (0..LEN).map(|_| rng.random()).collect();
    let b_vec: Vec<f32> = (0..LEN).map(|_| rng.random()).collect();

    group.bench_function("scalar_squared_euclidean", |b| {
        b.iter(|| {
            let sum = simd::squared_euclidean_scalar(black_box(&a), black_box(&b_vec));
            black_box(sum);
        });
    });

    group.bench_function("auto_vec_squared_euclidean", |b| {
        b.iter(|| {
            let sum = simd::squared_euclidean_auto_vec(black_box(&a), black_box(&b_vec));
            black_box(sum);
        });
    });

    #[cfg(target_arch = "aarch64")]
    group.bench_function("neon_squared_euclidean", |b| {
        b.iter(|| {
            let sum = simd::squared_euclidean_neon(black_box(&a), black_box(&b_vec))
                .unwrap_or_else(|| simd::squared_euclidean_scalar(&a, &b_vec));
            black_box(sum);
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("sse_squared_euclidean", |b| {
        b.iter(|| {
            let sum = simd::squared_euclidean_sse(black_box(&a), black_box(&b_vec))
                .unwrap_or_else(|| simd::squared_euclidean_scalar(&a, &b_vec));
            black_box(sum);
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("avx_squared_euclidean", |b| {
        b.iter(|| {
            let sum = simd::squared_euclidean_avx(black_box(&a), black_box(&b_vec))
                .unwrap_or_else(|| simd::squared_euclidean_scalar(&a, &b_vec));
            black_box(sum);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_brute_force_search, bench_simd_kernels);
criterion_main!(benches);
