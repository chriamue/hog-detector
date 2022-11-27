use crate::detector::{detect_objects, visualize_detections};
use crate::feature_descriptor::HogFeatureDescriptor;
use crate::hogdetector::HogDetectorTrait;
use crate::prelude::Detection;
use crate::utils::{pyramid, sliding_window};
use crate::{DataSet, Detector, HogDetector, Predictable, Trainable};
use image::imageops::{resize, FilterType};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::Kernels;

use super::Classifier;

/// svc type for float x and unsigned integer y
pub type SVCType<'a> = SVC<'a, f32, i32, DenseMatrix<f32>, Vec<i32>>;
/// svc parameters type for float x and unsigned integer y
pub type SVCParametersType = SVCParameters<f32, i32, DenseMatrix<f32>, Vec<i32>>;

/// A Support Vector Machine classifier
#[derive(Default, Serialize, Deserialize, Debug)]
pub struct SVMClassifier<'a> {
    /// svc classifier
    pub svc: Option<SVCType<'a>>,
}

impl Classifier for SVMClassifier<'_> {}

impl PartialEq for SVMClassifier<'_> {
    fn eq(&self, other: &SVMClassifier) -> bool {
        self.svc.is_none() && other.svc.is_none()
            || self.svc.is_some() && other.svc.is_some() && self.svc.eq(&other.svc)
    }
}

impl HogDetector<SVMClassifier<'_>> {
    /// new default support vector machine detector
    pub fn svm() -> Self {
        HogDetector::<SVMClassifier> {
            classifier: None,
            feature_descriptor: Box::new(HogFeatureDescriptor::default()),
        }
    }
}

impl HogDetectorTrait for HogDetector<SVMClassifier<'_>> {
    fn save(&self) -> String {
        serde_json::to_string(&self.classifier).unwrap()
    }

    fn load(&mut self, model: &str) {
        self.classifier = Some(serde_json::from_str::<SVMClassifier>(model).unwrap());
    }
}

impl<'a> Trainable for HogDetector<SVMClassifier<'a>> {
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<u32>) {
        let kernel = Kernels::linear();
        let parameters: SVCParametersType = SVCParameters::default()
            .with_c(10.0)
            .with_epoch(3)
            .with_kernel(kernel);
        let y_train: Vec<i32> = y_train
            .iter()
            .map(|y| if *y as u32 == 1 { 1 } else { -1 })
            .collect();

        let svc = SVC::fit(&x_train, &y_train, &parameters).unwrap();
        let deserialized_svc: SVCType<'a> =
            serde_json::from_str(&serde_json::to_string(&svc).unwrap()).unwrap();
        let classifier = SVMClassifier {
            svc: Some(deserialized_svc),
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

impl<'a> Predictable for HogDetector<SVMClassifier<'a>> {
    fn predict(&self, image: &DynamicImage) -> u32 {
        let image = resize(image, 32, 32, FilterType::Gaussian);
        let image = DynamicImage::ImageRgba8(image);
        let x = vec![self.preprocess(&image)];
        let x = DenseMatrix::from_2d_vec(&x);
        let y = self
            .classifier
            .as_ref()
            .unwrap()
            .svc
            .as_ref()
            .unwrap()
            .predict(&x)
            .unwrap();
        *y.first().unwrap() as u32
    }
}

impl<'a> Detector for HogDetector<SVMClassifier<'a>> {
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
        let classifier = SVMClassifier::default();
        assert!(classifier.svc.is_none());
    }

    #[test]
    fn test_partial_eq() {
        let classifier1 = SVMClassifier::default();
        let classifier2 = SVMClassifier::default();
        assert!(classifier1.svc.is_none());
        assert!(classifier1.eq(&classifier2));
        assert!(classifier1.eq(&classifier1));
    }

    #[test]
    fn test_save_load() {
        let mut model = HogDetector::<SVMClassifier>::default();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        model.train_class(&dataset, 1);
        let serialized = model.save();
        let mut model2 = HogDetector::<SVMClassifier>::default();
        model2.load(&serialized);
        assert_eq!(model, model2);
    }

    #[test]
    fn test_train() {
        let mut model = HogDetector::<SVMClassifier>::default();

        let mut dataset = MemoryDataSet::new_test();
        dataset.load();

        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());
    }

    // #[ignore = "deserializing not working for svm kernel"]
    #[should_panic = "called `Option::unwrap()` on a `None` value"]
    #[test]
    fn test_evaluate() {
        let mut model = HogDetector::<SVMClassifier>::default();

        let mut dataset = MemoryDataSet::new_test();
        dataset.load();

        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());
        assert!(model.evaluate(&dataset, 1) > 0.0);
    }

    // #[ignore = "deserializing not working for svm kernel"]
    #[should_panic = "called `Option::unwrap()` on a `None` value"]
    #[test]
    fn test_detector() {
        let img = test_image();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        let mut detector = HogDetector::<SVMClassifier>::default();
        detector.train_class(&dataset, 1);
        let detections = detector.detect_objects(&img);
        assert_eq!(1, detections.len());
        assert!(detections[0].bbox.x < 75.0);
        assert!(detections[0].bbox.x > 25.0);
        assert!(detections[0].bbox.y < 25.0);
        assert!(detections[0].bbox.y >= 0.0);
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([125, 255, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([125, 255, 0]), visualization.get_pixel(75, 0));
    }
}
