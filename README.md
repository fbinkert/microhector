# Microhector

A extremely minimal, in-memory vector database written in under 30 lines of Rust.

`microhector` is an educational project for demonstrating the core mechanics of vector search.

Currently, it implements a brute-force (exact match) Nearest Neighbor search using Euclidean distance.

## Quick Start

To use `microhector`, simply initialize a database with your required dimensionality, insert your vectors, and query them.

```rust
use microhector::VectorDatabase;

fn main() {
    // Initialize a 3-dimensional database
    let mut db = VectorDatabase::<3>::default();

    // Insert mock embeddings
    db.insert(1, [0.1, 0.2, 0.3]); // "Apple"
    db.insert(2, [0.9, 0.8, 0.7]); // "Car"
    db.insert(3, [0.15, 0.25, 0.35]); // "Banana"

    // Search for the closest matches to a new query vector
    let query = [0.12, 0.22, 0.32];
    let top_k = 2;
    let results = db.search(&query, top_k);

    println!("Top {} matches for query {:?}:", top_k, query);
    for res in results {
        println!("ID: {}, Distance: {:.4}", res.id, res.distance);
    }
}
