use crate::{
    detector::{detect_objects, visualize_detections},
    feature_descriptor::HogFeatureDescriptor,
    hogdetector::HogDetectorTrait,
    prelude::Detection,
    utils::{pyramid, sliding_window},
    DataSet, Detector, HogDetector, Predictable, Trainable,
};
use image::{
    imageops::{resize, FilterType},
    DynamicImage,
};
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier as RFC;
use smartcore::{linalg::basic::matrix::DenseMatrix, naive_bayes::gaussian::GaussianNB};

use super::{BayesClassifier, Classifier, RandomForestClassifier};
/// A naive bayes classifier
#[derive(Default, Serialize, Deserialize, Debug)]
pub struct CombinedClassifier {
    /// inner bayes
    pub bayes: BayesClassifier,
    /// inner random forest
    pub randomforest: RandomForestClassifier,
}

impl Classifier for CombinedClassifier {}

impl PartialEq for CombinedClassifier {
    fn eq(&self, other: &CombinedClassifier) -> bool {
        self.bayes.eq(&other.bayes) && self.randomforest.eq(&other.randomforest)
    }
}

impl HogDetector<CombinedClassifier> {
    /// new default combined
    pub fn combined() -> Self {
        HogDetector::<CombinedClassifier> {
            classifier: None,
            feature_descriptor: Box::new(HogFeatureDescriptor::default()),
        }
    }
}

impl HogDetectorTrait for HogDetector<CombinedClassifier> {
    fn save(&self) -> String {
        serde_json::to_string(&self.classifier).unwrap()
    }

    fn load(&mut self, model: &str) {
        self.classifier = Some(serde_json::from_str::<CombinedClassifier>(model).unwrap());
    }
}

impl Trainable for HogDetector<CombinedClassifier> {
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<u32>) {
        let nb = GaussianNB::fit(&x_train, &y_train, Default::default()).unwrap();
        let bayes = BayesClassifier { inner: Some(nb) };
        let rfc = RFC::fit(&x_train, &y_train, Default::default()).unwrap();
        let randomforest = RandomForestClassifier { inner: Some(rfc) };
        let classifier = CombinedClassifier {
            bayes,
            randomforest,
        };
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

impl Predictable for HogDetector<CombinedClassifier> {
    fn predict(&self, image: &DynamicImage) -> u32 {
        let image = resize(image, 32, 32, FilterType::Gaussian);
        let image = DynamicImage::ImageRgba8(image);
        let x = vec![self.preprocess(&image)];
        let x = DenseMatrix::from_2d_vec(&x);
        let bayes_y = *self
            .classifier
            .as_ref()
            .unwrap()
            .bayes
            .inner
            .as_ref()
            .unwrap()
            .predict(&x)
            .unwrap_or_else(|_| vec![0])
            .first()
            .unwrap();
        let randomforest_y = *self
            .classifier
            .as_ref()
            .unwrap()
            .randomforest
            .inner
            .as_ref()
            .unwrap()
            .predict(&x)
            .unwrap_or_else(|_| vec![0])
            .first()
            .unwrap();
        if bayes_y == randomforest_y {
            bayes_y
        } else {
            0
        }
    }
}

impl Detector for HogDetector<CombinedClassifier> {
    fn detect_objects(&self, image: &DynamicImage) -> Vec<Detection> {
        let step_size = 8;
        let window_size = 32;
        let mut windows = sliding_window(image, step_size, window_size);
        windows.extend(pyramid(image, 1.3, step_size, window_size));
        windows.extend(pyramid(image, 1.5, step_size, window_size));

        let predictions: Vec<(u32, u32, u32)> = windows
            .iter()
            .map(|(x, y, window)| (*x, *y, self.predict(window)))
            .collect();
        detect_objects(predictions, window_size)
    }

    fn visualize_detections(&self, image: &DynamicImage) -> DynamicImage {
        let detections = self.detect_objects(image);
        visualize_detections(image, &detections)
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use crate::{dataset::MemoryDataSet, tests::test_image};

    use super::*;

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
        let mut detector = HogDetector::<BayesClassifier>::default();
        detector.train_class(&dataset, 1);
        let detections = detector.detect_objects(&img);
        assert!(detections.is_empty());
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([0, 0, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([255, 0, 0]), visualization.get_pixel(75, 0));
    }
}
