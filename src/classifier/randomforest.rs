use crate::{
    feature_descriptor::HogFeatureDescriptor, hogdetector::HogDetectorTrait, DataSet, Detector,
    HogDetector, Predictable, Trainable,
};
use image::{
    imageops::{resize, FilterType},
    DynamicImage,
};
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier as RFC;
use smartcore::linalg::basic::matrix::DenseMatrix;

use super::Classifier;
type RFCType = RFC<f32, u32, DenseMatrix<f32>, Vec<u32>>;

/// A Random Forest classifier
#[derive(Default, Serialize, Deserialize, Debug)]
pub struct RandomForestClassifier {
    /// inner
    pub inner: Option<RFCType>,
}

impl Classifier for RandomForestClassifier {}

impl PartialEq for RandomForestClassifier {
    fn eq(&self, other: &RandomForestClassifier) -> bool {
        self.inner.is_none() && other.inner.is_none()
            || self.inner.is_some() && other.inner.is_some()
    }
}

impl HogDetector<RandomForestClassifier> {
    /// new default random forest
    pub fn random_forest() -> Self {
        HogDetector::<RandomForestClassifier> {
            classifier: None,
            feature_descriptor: Box::new(HogFeatureDescriptor::default()),
        }
    }
}

impl HogDetectorTrait for HogDetector<RandomForestClassifier> {
    fn save(&self) -> String {
        serde_json::to_string(&self.classifier).unwrap()
    }

    fn load(&mut self, model: &str) {
        self.classifier = Some(serde_json::from_str::<RandomForestClassifier>(model).unwrap());
    }

    fn detector(&self) -> &dyn Detector {
        self
    }
}

impl Trainable for HogDetector<RandomForestClassifier> {
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<u32>) {
        let rfc = RFC::fit(&x_train, &y_train, Default::default()).unwrap();
        let classifier = RandomForestClassifier { inner: Some(rfc) };
        self.classifier = Some(classifier);
    }

    fn train_class(&mut self, dataset: &dyn DataSet, class: u32) {
        let (x_train, y_train, _, _) = dataset.get();
        let x_train = self.preprocess_matrix(x_train);
        let y_train = y_train
            .iter()
            .map(|y| if *y as u32 == class { *y } else { 0u32 })
            .collect();
        self.train(x_train, y_train);
    }
    fn evaluate(&mut self, dataset: &dyn DataSet, class: u32) -> f32 {
        let mut i = 0;
        let (x_train, y_train, _, _) = dataset.get();
        x_train.iter().zip(y_train).for_each(|(img, y)| {
            let pred = self.predict(img);
            if (pred == y && y == class) || (pred == 0 && y != class) {
                i += 1;
            }
        });
        i as f32 / x_train.len() as f32
    }
}

impl Predictable for HogDetector<RandomForestClassifier> {
    fn predict(&self, image: &DynamicImage) -> u32 {
        let image = resize(image, 32, 32, FilterType::Gaussian);
        let image = DynamicImage::ImageRgba8(image);
        let x = vec![self.preprocess(&image)];
        let x = DenseMatrix::from_2d_vec(&x);
        let y = self
            .classifier
            .as_ref()
            .unwrap()
            .inner
            .as_ref()
            .unwrap()
            .predict(&x)
            .unwrap();
        *y.first().unwrap() as u32
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use crate::{dataset::MemoryDataSet, tests::test_image};

    use super::*;

    #[test]
    fn test_default() {
        let classifier = RandomForestClassifier::default();
        assert!(classifier.inner.is_none());
    }

    #[test]
    fn test_partial_eq() {
        let detector1 = HogDetector::default();
        let detector2 = HogDetector::random_forest();
        assert!(detector1.eq(&detector2));
    }

    #[test]
    fn test_save_load() {
        let mut model = HogDetector::<RandomForestClassifier>::default();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        model.train_class(&dataset, 1);
        let serialized = model.save();
        let mut model2 = HogDetector::<RandomForestClassifier>::default();
        model2.load(&serialized);
        assert_eq!(model, model2);
    }

    #[test]
    fn test_evaluate() {
        let mut model = HogDetector::<RandomForestClassifier>::default();

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
        let mut detector = HogDetector::<RandomForestClassifier>::default();
        detector.train_class(&dataset, 1);
        let detections = detector.detect_objects(&img);
        assert!(!detections.is_empty());
        assert!(detections[0].bbox.x < 75.0);
        assert!(detections[0].bbox.x > 25.0);
        assert!(detections[0].bbox.y < 25.0);
        assert!(detections[0].bbox.y >= 0.0);
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([125, 255, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([125, 255, 0]), visualization.get_pixel(75, 0));
    }
}
