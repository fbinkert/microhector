/// Represents a single search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityResult {
    pub id: u64,
    pub distance: f32,
}

#[derive(Default)]
pub struct VectorDatabase<const DIM: usize>(Vec<(u64, [f32; DIM])>);

impl<const DIM: usize> VectorDatabase<DIM> {
    /// Insert a vector with an ID.
    pub fn insert(&mut self, id: u64, vector: [f32; DIM]) {
        self.0.push((id, vector));
    }

    /// Search for the top K nearest neighbors.
    #[must_use]
    pub fn search(&self, query: &[f32; DIM], k: usize) -> Vec<SimilarityResult> {
        let mut result: Vec<SimilarityResult> = self
            .0
            .iter()
            .map(|(id, vector)| SimilarityResult {
                id: *id,
                // Squared Euclidean distance
                distance: query.iter().zip(vector).map(|(a, b)| (a - b).powi(2)).sum(),
            })
            .collect();

        result.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        result.truncate(k);
        result
    }
}
