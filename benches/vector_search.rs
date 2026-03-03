use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use microhector::{VectorDatabase, VectorDatabaseSoA};
use rand::RngExt;

const DIM: usize = 128;

fn bench_brute_force_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    group.sample_size(60);

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

criterion_group!(benches, bench_brute_force_search);
criterion_main!(benches);
