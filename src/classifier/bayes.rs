use crate::{hogdetector::HogDetectorTrait, Detector, HogDetector};
use image::DynamicImage;
use linfa::Float;
use linfa::Label;
use ndarray::{Array1, ArrayView2};
use object_detector_rust::{
    prelude::{HOGFeature, Predictable},
    trainable::Trainable,
    utils::{SlidingWindow, WindowGenerator},
};
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::naive_bayes::gaussian::GaussianNB;

use super::Classifier;
type BayesType = GaussianNB<f32, u32, DenseMatrix<f32>, Vec<u32>>;

/// A naive bayes classifier
#[derive(Default, Serialize, Deserialize, Debug)]
pub struct BayesClassifier {
    /// inner
    pub inner: Option<BayesType>,
}

impl PartialEq for BayesClassifier {
    fn eq(&self, other: &BayesClassifier) -> bool {
        self.inner.is_none() && other.inner.is_none()
            || self.inner.is_some() && other.inner.is_some()
    }
}

impl<X, Y> HogDetector<X, Y, BayesClassifier, SlidingWindow>
where
    X: Float,
    Y: Label,
{
    /// new default bayes
    pub fn bayes() -> Self {
        HogDetector::<X, Y, BayesClassifier, SlidingWindow> {
            classifier: None,
            feature_descriptor: Box::new(HOGFeature::default()),
            window_generator: SlidingWindow {
                width: 32,
                height: 32,
                step_size: 32,
            },
            x: std::marker::PhantomData,
            y: std::marker::PhantomData,
        }
    }
}

impl<X, Y, W> HogDetectorTrait<X, Y> for HogDetector<X, Y, BayesClassifier, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    fn save(&self) -> String {
        serde_json::to_string(&self.classifier).unwrap()
    }

    fn load(&mut self, model: &str) {
        self.classifier = Some(serde_json::from_str::<BayesClassifier>(model).unwrap());
    }

    fn detector(&self) -> &dyn Detector {
        self
    }
}

impl<X, Y> Trainable<X, Y> for BayesClassifier
where
    X: Float,
    Y: Label,
{
    fn fit(
        &mut self,
        x: &ndarray::ArrayView2<X>,
        y: &ndarray::ArrayView1<Y>,
    ) -> Result<(), String> {
        let nb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
        let classifier = BayesClassifier { inner: Some(nb) };
        self.classifier = Some(classifier);
        Ok(())
    }
}

impl<X, Y> Predictable<X, Y> for BayesClassifier
where
    X: Float,
    Y: Label,
{
    fn predict(&self, x: &ArrayView2<X>) -> Result<Array1<Y>, String> {
        Ok(self.model.as_ref().unwrap().predict(x))
    }
}

impl<X, Y> Classifier<X, Y> for BayesClassifier
where
    X: Float,
    Y: Label,
{
}

#[cfg(test)]
mod tests {
    use image::Rgb;
    use object_detector_rust::{prelude::MemoryDataSet, tests::test_image};

    use super::*;

    #[test]
    fn test_default() {
        let classifier = BayesClassifier::default();
        assert!(classifier.inner.is_none());
    }

    #[test]
    fn test_partial_eq() {
        let detector1 = HogDetector::default();
        let detector2 = HogDetector::bayes();
        assert!(detector1.eq(&detector2));
    }

    #[test]
    fn test_save_load() {
        let mut model = HogDetector::<BayesClassifier>::default();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        model.train_class(&dataset, 1);
        let serialized = model.save();
        let mut model2 = HogDetector::<BayesClassifier>::default();
        model2.load(&serialized);
        assert_eq!(model, model2);
    }

    #[test]
    fn test_evaluate() {
        let mut model = HogDetector::<BayesClassifier>::default();

        let mut dataset = MemoryDataSet::new_test();
        dataset.load();

        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());
        assert!(model.evaluate(&dataset, 1) > 0.0);
    }

    #[test]
    fn test_detector() {
        let img = test_image();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        let mut detector = HogDetector::<BayesClassifier>::default();
        detector.train_class(&dataset, 1);
        let detections = detector.detect_objects(&img);
        assert!(detections.is_empty());
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([0, 0, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([255, 0, 0]), visualization.get_pixel(75, 0));
    }
}
