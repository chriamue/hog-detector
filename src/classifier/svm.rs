use std::fmt::Debug;

use ndarray::{Array1, ArrayView1, ArrayView2};
use object_detector_rust::{
    classifier::Classifier, predictable::Predictable, trainable::Trainable,
    window_generator::PyramidWindow,
};
use serde::{Serialize, Deserialize};
use svm_burns::{svm::SVM, Parameters, RBFKernel, SVC};

use crate::HogDetector;

/// A support vector machine classifier
#[derive(Default, Serialize, Deserialize)]
pub struct SVMClassifier {
    model: Option<SVC>,
}

impl Debug for SVMClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SVMClassifier").finish()
    }
}

impl SVMClassifier {
    /// Creates a new `SVMClassifier`
    pub fn new() -> Self {
        SVMClassifier { model: None }
    }
}

impl PartialEq for SVMClassifier {
    fn eq(&self, other: &SVMClassifier) -> bool {
        self.model.is_none() && other.model.is_none()
            || self.model.is_some() && other.model.is_some()
    }
}

impl HogDetector<f32, usize, SVMClassifier, PyramidWindow> {
    /// new default svm detector
    pub fn svm() -> Self {
        HogDetector::<f32, usize, SVMClassifier, PyramidWindow>::default()
    }
}

impl Trainable<f32, usize> for SVMClassifier {
    fn fit(&mut self, x: &ArrayView2<f32>, y: &ArrayView1<usize>) -> Result<(), String> {
        let x_vec: Vec<Vec<f64>> = x
            .outer_iter()
            .map(|row| row.iter().map(|&elem| elem as f64).collect())
            .collect();

        let y_vec: Vec<i32> = y.iter().map(|&elem| elem as i32).collect();

        let mut parameters = Parameters::default();
        parameters.with_kernel(Box::new(RBFKernel::new(0.7)));
        let mut svc = SVC::new(parameters);

        svc.fit(&x_vec, &y_vec);
        self.model = Some(svc);
        Ok(())
    }
}

impl Predictable<f32, usize> for SVMClassifier {
    fn predict(&self, x: &ArrayView2<f32>) -> Result<Array1<usize>, String> {
        let x_vec: Vec<Vec<f64>> = x
            .outer_iter()
            .map(|row| row.iter().map(|&elem| elem as f64).collect())
            .collect();
        let prediction = self.model.as_ref().unwrap().predict(&x_vec);
        let prediction: Vec<usize> = prediction
            .iter()
            .map(|&x| if x > 0 { 1 } else { 0 })
            .collect();
        Ok(Array1::from(prediction))
    }
}

impl Classifier<f32, usize> for SVMClassifier {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hogdetector::HogDetectorTrait;
    use image::Rgb;
    use object_detector_rust::dataset::DataSet;
    use object_detector_rust::detector::Detector;
    use object_detector_rust::{prelude::MemoryDataSet, tests::test_image};

    #[test]
    fn test_default() {
        let classifier = super::SVMClassifier::default();
        assert!(classifier.model.is_none());
    }

    #[test]
    fn test_partial_eq() {
        let detector1 = HogDetector::<f32, usize, super::SVMClassifier, _>::default();
        let detector2 = HogDetector::<f32, usize, super::SVMClassifier, _>::svm();
        assert!(detector1.eq(&detector2));
    }

    #[test]
    fn test_detector() {
        let img = test_image();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load().unwrap();
        let (x, y) = dataset.get_data();
        let x = x.into_iter().map(|x| x.thumbnail_exact(32, 32)).collect();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();

        let mut detector: HogDetector<f32, usize, super::SVMClassifier, _> = HogDetector::default();
        detector.fit_class(&x, &y, 1).unwrap();
        let detections = detector.detect(&img);
        assert!(detections.is_empty());
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([0, 0, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([255, 0, 0]), visualization.get_pixel(75, 0));
    }
}
