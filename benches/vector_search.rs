use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use microhector::VectorDatabase;
use rand::RngExt;

const DIM: usize = 128;

fn bench_brute_force_search(c: &mut Criterion) {
    let mut db = VectorDatabase::<DIM>::default();
    let mut rng = rand::rng();
    let num_records = 10_000u64;

    // Populate the database
    for i in 0..num_records {
        let vector: [f32; DIM] = std::array::from_fn(|_| rng.random());
        db.insert(i, vector);
    }

    // Create a random query vector
    let query: [f32; DIM] = std::array::from_fn(|_| rng.random());
    let top_k = 10;

    // Run the benchmark
    c.bench_function("search_10k_records_dim128", |b| {
        b.iter(|| {
            let results = db.search(black_box(&query), black_box(top_k));
            black_box(results);
        });
    });
}

criterion_group!(benches, bench_brute_force_search);
criterion_main!(benches);
