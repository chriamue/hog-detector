use crate::HogDetector;
use linfa::Float;
use linfa::Label;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::{Array1, ArrayView2};
use num_traits::Unsigned;
use object_detector_rust::{
    prelude::{Predictable, PyramidWindow},
    trainable::Trainable,
};
use serde::{Deserialize, Serialize};
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::numbers::basenum::Number;
use smartcore::numbers::realnum::RealNumber;

use super::Classifier;

/// A naive bayes classifier
#[derive(Default, Serialize, Deserialize, Debug)]
pub struct BayesClassifier<X, Y>
where
    X: Float + Number + RealNumber,
    Y: Label + Number + Ord + Unsigned,
{
    model: Option<GaussianNB<X, Y, Array2<X>, Array1<Y>>>,
}

impl<X, Y> BayesClassifier<X, Y>
where
    X: Float + Number + RealNumber,
    Y: Label + Number + Ord + Unsigned,
{
    /// Creates a new `BayesClassifier` instance with default parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use hog_detector::classifier::BayesClassifier;
    ///
    /// let bayes_classifier = BayesClassifier::<f32, usize>::new();
    /// ```
    pub fn new() -> Self {
        BayesClassifier { model: None }
    }
}

impl<X, Y> PartialEq for BayesClassifier<X, Y>
where
    X: Float + Number + RealNumber,
    Y: Label + Number + Ord + Unsigned,
{
    fn eq(&self, other: &BayesClassifier<X, Y>) -> bool {
        self.model.is_none() && other.model.is_none()
            || self.model.is_some() && other.model.is_some()
    }
}

impl<X, Y> HogDetector<X, Y, BayesClassifier<X, Y>, PyramidWindow>
where
    X: Float + Number + RealNumber,
    Y: Label + Number + Ord + Unsigned,
{
    /// new default bayes
    pub fn bayes() -> Self {
        HogDetector::<X, Y, BayesClassifier<X, Y>, PyramidWindow>::default()
    }
}

impl<X, Y> Trainable<X, Y> for BayesClassifier<X, Y>
where
    X: Float + Number + RealNumber,
    Y: Label + Number + Ord + Unsigned,
{
    fn fit(&mut self, x: &ArrayView2<X>, y: &ArrayView1<Y>) -> Result<(), String> {
        let x = x.to_owned();
        let y = y.to_owned();
        let nb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
        self.model = Some(nb);
        Ok(())
    }
}

impl<X, Y> Predictable<X, Y> for BayesClassifier<X, Y>
where
    X: Float + Number + RealNumber,
    Y: Label + Number + Ord + Unsigned,
{
    fn predict(&self, x: &ArrayView2<X>) -> Result<Array1<Y>, String> {
        match self.model.as_ref().unwrap().predict(&x.to_owned()) {
            Ok(prediction) => Ok(prediction),
            _ => Ok(Array1::default(x.dim().0)),
        }
    }
}

impl<X, Y> Classifier<X, Y> for BayesClassifier<X, Y>
where
    X: Float + Number + RealNumber,
    Y: Label + Number + Ord + Unsigned,
{
}

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
        let classifier = super::BayesClassifier::<f32, usize>::default();
        assert!(classifier.model.is_none());
    }

    #[test]
    fn test_partial_eq() {
        let detector1 = HogDetector::<f32, usize, super::BayesClassifier<_, _>, _>::default();
        let detector2 = HogDetector::<f32, usize, super::BayesClassifier<_, _>, _>::bayes();
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

        let mut detector: HogDetector<f32, usize, super::BayesClassifier<_, _>, _> =
            HogDetector::default();
        detector.fit_class(&x, &y, 1).unwrap();
        let detections = detector.detect(&img);
        assert!(detections.is_empty());
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([0, 0, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([255, 0, 0]), visualization.get_pixel(75, 0));
    }
}
