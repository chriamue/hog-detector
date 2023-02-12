//! This module contains filter to eliminate false positive detections.

/// Defines a trait for filtering detections to eliminate false positive detections.
pub trait DetectionFilter {
    /// Filters a set of detections and returns a filtered set of detections, with false positive detections removed.
    ///
    /// # Arguments
    /// * `detections` - A vector of `Detection` objects representing the set of detections to be filtered.
    ///
    /// # Returns
    /// A vector of `Detection` objects representing the filtered set of detections, with false positive detections removed.
    fn filter_detections(&self, detections: &Vec<Detection>) -> Vec<Detection>;
}
