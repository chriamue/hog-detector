use super::DetectionFilter;
use crate::prelude::Detection;

/// `TrackerFilter` is a struct for tracking and filtering object detections over time.
/// It refines the detections by comparing the overlap between the previous frame's detections
/// and the current frame's detections, using an overlap threshold to filter out false positives.
///
/// # Examples
///
/// ```
/// use object_detector_rust::{Detection, TrackerFilter};
///
/// let mut tracker = TrackerFilter {
///     previous_detections: Vec::new(),
///     overlap_threshold: 0.5,
/// };
///
/// let current_detections: Vec<Detection> = /* ... */;
///
/// let refined_detections = tracker.filter_and_update(current_detections);
/// ```
///
/// This example demonstrates how to create a `TrackerFilter` instance and use it to filter and
/// refine object detections in a frame.
#[derive(Debug, Default)]
pub struct TrackerFilter {
    /// The previous frame's detections
    pub previous_detections: Vec<Detection>,
    /// The overlap threshold for filtering out false positives
    pub overlap_threshold: f32,
}

impl DetectionFilter for TrackerFilter {
    fn filter_detections(&mut self, detections: &Vec<Detection>) -> Vec<Detection> {
        let mut filtered_detections = Vec::new();

        for detection in detections {
            for prev_detection in &self.previous_detections {
                if prev_detection.class == detection.class
                    && prev_detection.bbox.overlap(&detection.bbox) > self.overlap_threshold
                {
                    filtered_detections.push(detection.clone());
                    break;
                }
            }
        }

        self.previous_detections = detections.clone();
        filtered_detections
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_detector_rust::prelude::BBox;

    #[test]
    fn test_tracker_filter() {
        let detections = vec![
            Detection::new(BBox::new(0, 0, 100, 100), 1, 0.9),
            Detection::new(BBox::new(50, 0, 100, 100), 1, 0.8),
        ];
        let mut tracker = TrackerFilter::default();
        assert_eq!(tracker.filter_detections(&detections).len(), 0);

        let new_detections = vec![
            Detection::new(BBox::new(0, 0, 100, 100), 1, 0.95),
            Detection::new(BBox::new(50, 0, 100, 100), 1, 0.85),
        ];
        assert_eq!(tracker.filter_detections(&new_detections).len(), 2);
    }
}
