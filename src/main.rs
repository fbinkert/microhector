use microhector::VectorDatabase;

fn main() {
    let mut db = VectorDatabase::<3>::default();

    // Insert some mock embeddings
    db.insert(1, [0.1, 0.2, 0.3]); // Apple
    db.insert(2, [0.9, 0.8, 0.7]); // Car
    db.insert(3, [0.15, 0.25, 0.35]); // Banana

    // Search for a vector similar to "Apple"
    let query = [0.12, 0.22, 0.32];
    let results = db.search(&query, 2);

    println!("Top 2 matches for query {query:?}:");
    for res in results {
        println!("ID: {}, Distance: {:.4}", res.id, res.distance);
    }
}
