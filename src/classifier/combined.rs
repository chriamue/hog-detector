use crate::{hogdetector::HogDetectorTrait, Detector, HogDetector};
use image::{
    imageops::{resize, FilterType},
    DynamicImage,
};
use linfa::{Dataset, Float, Label};
use ndarray::{Array1, ArrayView1, ArrayView2};
use object_detector_rust::{
    prelude::{HOGFeature, Predictable, RandomForestClassifier},
    trainable::Trainable,
    utils::{SlidingWindow, WindowGenerator},
};
use serde::{Deserialize, Serialize};
use smartcore::{linalg::basic::matrix::DenseMatrix, numbers::{floatnum::FloatNumber, basenum::Number}};

use super::BayesClassifier;
/// A naive bayes classifier
#[derive(Default, Serialize, Deserialize, Debug)]
pub struct CombinedClassifier<X, Y>
where
    X: Float + FloatNumber,
    Y: Label + Number,
{
    /// inner bayes
    pub bayes: BayesClassifier,
    /// inner random forest
    pub randomforest: RandomForestClassifier<X, Y>,
}

impl<X, Y> PartialEq for CombinedClassifier<X, Y>
where
    X: Float + FloatNumber,
    Y: Label + Number,
{
    fn eq(&self, other: &CombinedClassifier<X, Y>) -> bool {
        self.bayes.eq(&other.bayes) && self.randomforest.eq(&other.randomforest)
    }
}

impl<X, Y> HogDetector<X, Y, CombinedClassifier<X, Y>, SlidingWindow>
where
    X: Float,
    Y: Label,
{
    /// new default combined
    pub fn combined() -> Self {
        HogDetector::<CombinedClassifier<X, Y>, SlidingWindow> {
            classifier: None,
            feature_descriptor: Box::new(HOGFeature::default()),
            window_generator: SlidingWindow {
                width: 32,
                height: 32,
                step_size: 32,
            },
        }
    }
}

impl<X, Y, W> HogDetectorTrait<X, Y> for HogDetector<X, Y, CombinedClassifier<X, Y>, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    fn save(&self) -> String {
        serde_json::to_string(&self.classifier).unwrap()
    }

    fn load(&mut self, model: &str) {
        self.classifier = Some(serde_json::from_str::<CombinedClassifier<X, Y>>(model).unwrap());
    }

    fn detector(&self) -> &dyn Detector {
        self
    }
}

impl<X, Y, W> Trainable<X, Y> for HogDetector<X, Y, CombinedClassifier<X, Y>, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    fn fit(&mut self, x: &ArrayView2<X>, y: &ArrayView1<Y>) -> Result<(), String> {
        let dataset = Dataset::new(x.to_owned(), y.to_owned());
        let mut bayes = BayesClassifier::default();
        bayes.fit(&x, &y);
        let mut randomforest = RandomForestClassifier::<X, Y>::default();
        randomforest.fit(&x, &y).unwrap();
        let classifier = CombinedClassifier {
            bayes,
            randomforest,
        };
        self.classifier = Some(classifier);
        Ok(())
    }
}

impl<X, Y, W> Predictable<X, Y> for HogDetector<X, Y, CombinedClassifier<X, Y>, W>
where
    X: Float + FloatNumber,
    Y: Label + Number,
    W: WindowGenerator<DynamicImage>,
{
    fn predict(&self, x: &ArrayView2<X>) -> Result<Array1<Y>, String> {
        let bayes_y = self
            .classifier
            .as_ref()
            .unwrap()
            .bayes
            .as_ref()
            .unwrap()
            .predict(&x)
            .unwrap();
        let randomforest_y = self
            .classifier
            .as_ref()
            .unwrap()
            .randomforest
            .as_ref()
            .unwrap()
            .predict(&x)
            .unwrap();
        let combined = bayes_y
            .into_iter()
            .zip(randomforest_y)
            .map(|(a, b)| if a == b { a } else { 0 })
            .collect();
        Ok(combined)
    }
}

impl<X, Y> Classifier<X, Y> for CombinedClassifier<X, Y>
where
    X: Float + Number + RealNumber,
    Y: Label + Number + Ord + Unsigned,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use object_detector_rust::{prelude::MemoryDataSet, tests::test_image};

    #[test]
    fn test_default() {
        let _classifier = CombinedClassifier::default();
    }

    #[test]
    fn test_partial_eq() {
        let detector1 = HogDetector::default();
        let detector2 = HogDetector::combined();
        assert!(detector1.eq(&detector2));
    }

    #[test]
    fn test_save_load() {
        let mut model = HogDetector::<CombinedClassifier>::default();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        model.train_class(&dataset, 1);
        let serialized = model.save();
        let mut model2 = HogDetector::<CombinedClassifier>::default();
        model2.load(&serialized);
        assert_eq!(model, model2);
    }

    #[test]
    fn test_evaluate() {
        let mut model = HogDetector::<CombinedClassifier>::default();

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
        let mut detector = HogDetector::<CombinedClassifier>::default();
        detector.train_class(&dataset, 1);
        let detections = detector.detect_objects(&img);
        assert!(detections.is_empty());
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([0, 0, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([255, 0, 0]), visualization.get_pixel(75, 0));
    }
}
