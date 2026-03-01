use std::cmp::Ordering;

/// Represents a single search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityResult {
    pub id: u64,
    pub distance: f32,
}

pub struct VectorDatabase<const DIM: usize> {
    records: Vec<(u64, [f32; DIM])>,
}

impl<const DIM: usize> VectorDatabase<DIM> {
    /// Insert a vecotr with an ID.
    pub fn insert(&mut self, id: u64, vector: [f32; DIM]) {
        self.records.push((id, vector));
    }

    /// Search for the top K nearest neighbors.
    #[must_use]
    pub fn search(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        let mut result: Vec<SimilarityResult> = self
            .records
            .iter()
            .map(|(id, vector)| SimilarityResult {
                id: *id,
                distance: euclidean_distance(query, vector),
            })
            .collect();

        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result.truncate(k);
        result
    }
}

impl<const DIM: usize> Default for VectorDatabase<DIM> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

/// Calculate the Euclidean distance between two vectors.
fn euclidean_distance<const DIM: usize>(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

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
