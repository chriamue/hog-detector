use super::DetectionFilter;
use object_detector_rust::prelude::Detection;

/// Tracker for filtering and refining object detections.
#[derive(Debug, Default)]
pub struct TrackerFilter {}

impl DetectionFilter for TrackerFilter {
    fn filter_detections(&self, detections: &Vec<Detection>) -> Vec<Detection> {
        detections.clone()
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
        let tracker = TrackerFilter::default();
        assert_eq!(tracker.filter_detections(&detections).len(), 2);
    }
}
