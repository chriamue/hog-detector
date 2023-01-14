use std::marker::PhantomData;

use crate::Detector;
use image::{DynamicImage, GenericImageView};
use linfa::{Float, Label};
use ndarray::Array2;
use object_detector_rust::prelude::{Feature, HOGFeature, Predictable};
use object_detector_rust::trainable::Trainable;
use object_detector_rust::{prelude::Classifier, utils::SlidingWindow};
use object_detector_rust::{
    prelude::{BBox, Class, Detection},
    utils::WindowGenerator,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Hog Detector struct
/// ,-----.  ,---------.   ,--------------------.   ,---.                             
/// |Image|  |GrayImage|   |HogFeatureDescriptor|   |SVM|                             
/// |-----|--|---------|---|--------------------|---|---|                             
/// `-----'  `---------'   `--------------------'   `---'                             
///                                                    |                              
///                                                    |                              
///                                             ,-----------.   ,--------.   ,-------.
///                                             |Predictions|   |Detector|   |Objects|
///                                             |-----------|---|--------|---|-------|
///                                             `-----------'   `--------'   `-------'
///
#[derive(Debug)]
pub struct HogDetector<X, Y, C: Classifier<X, Y>, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    x: PhantomData<*const X>,
    y: PhantomData<*const Y>,
    /// support vector classifier
    pub classifier: Option<C>,
    /// the feature descriptor
    pub feature_descriptor: Box<dyn Feature>,

    window_generator: W,
}

/// trait of an hog detector
pub trait HogDetectorTrait<X, Y>: Trainable<X, Y> + Detector + Send + Sync {
    /// save to string
    fn save(&self) -> String;
    /// load from string
    fn load(&mut self, model: &str);
    /// reference to detector trait
    fn detector(&self) -> &dyn Detector;
}

unsafe impl<X, Y, C: Classifier<X, Y>, W> Send for HogDetector<X, Y, C, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
}
unsafe impl<X, Y, C: Classifier<X, Y>, W> Sync for HogDetector<X, Y, C, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
}

impl<X, Y, C: Classifier<X, Y>, W> PartialEq for HogDetector<X, Y, C, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    fn eq(&self, other: &HogDetector<X, Y, C, W>) -> bool {
        self.classifier.is_none() && other.classifier.is_none()
            || self.classifier.is_some()
                && other.classifier.is_some()
                && self.classifier.eq(&other.classifier)
    }
}

impl<X, Y, C: Classifier<X, Y>> Default for HogDetector<X, Y, C, SlidingWindow>
where
    X: Float,
    Y: Label,
{
    fn default() -> Self {
        HogDetector::<X, Y, C, SlidingWindow> {
            classifier: None,
            feature_descriptor: Box::new(HOGFeature::default()),
            window_generator: SlidingWindow {
                width: 32,
                height: 32,
                step_size: 32,
            },
            x: PhantomData,
            y: PhantomData,
        }
    }
}

impl<X, Y, C: Classifier<X, Y>, W> HogDetector<X, Y, C, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    /// preprocesses image to vector
    pub fn preprocess(&self, image: &DynamicImage) -> Vec<f32> {
        self.feature_descriptor.extract(image).unwrap()
    }
    /// preprocesses images into dense matrix
    pub fn preprocess_matrix(&self, images: Vec<DynamicImage>) -> DenseMatrix<f32> {
        let descriptors: Vec<Vec<f32>> =
            images.iter().map(|image| self.preprocess(image)).collect();
        DenseMatrix::from_2d_vec(&descriptors)
    }
}

impl<C: Classifier<f32, usize>, W> Detector for HogDetector<f32, usize, C, W>
where
    HogDetector<f32, usize, C, W>: Predictable<f32, usize>,
    W: WindowGenerator<DynamicImage>,
{
    fn detect(&self, image: &DynamicImage) -> Vec<object_detector_rust::prelude::Detection> {
        let windows = self.window_generator.windows(image);
        let windows_len = windows.len();
        let hog_features: Vec<Vec<f32>> = windows
            .iter()
            .flat_map(|window| {
                self.feature_descriptor
                    .extract(&DynamicImage::ImageRgba8(window.view.to_image()))
            })
            .collect();

        let features_len = match hog_features.first() {
            Some(features) => features.len(),
            None => 0,
        };
        let hog_features: Vec<f32> = hog_features.into_iter().flatten().collect();
        let hog_features =
            Array2::from_shape_vec((windows_len, features_len), hog_features).unwrap();
        let predictions = self
            .classifier
            .as_ref()
            .unwrap()
            .predict(&hog_features.view())
            .unwrap();
        assert_eq!(windows_len, predictions.len());
        let mut detections = Vec::new();
        for (i, &prediction) in predictions.iter().enumerate() {
            if prediction > 0 {
                let window = windows[i];
                detections.push(Detection::new(
                    BBox::new(
                        window.x as i32,
                        window.y as i32,
                        window.view.width(),
                        window.view.height(),
                    ),
                    1 as Class,
                    1.0,
                ));
            }
        }
        detections
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops::resize, imageops::FilterType, open};
    use object_detector_rust::prelude::SVMClassifier;

    #[test]
    fn test_default() {
        let model = HogDetector::<SVMClassifier, SlidingWindow>::default();
        assert!(model.classifier.is_none());
    }

    #[test]
    fn test_part_eq() {
        let model1 = HogDetector::<SVMClassifier, SlidingWindow>::default();
        let model2 = HogDetector::<SVMClassifier, SlidingWindow>::default();
        assert!(model1.classifier.is_none());
        assert!(model1.eq(&model2));
        assert!(model1.eq(&model1));
    }

    #[test]
    fn test_hog() {
        let model = HogDetector::<SVMClassifier, SlidingWindow>::default();
        let loco03 = open("res/loco03.jpg").unwrap().to_rgb8();
        let loco03 = resize(&loco03, 32, 32, FilterType::Nearest);
        let descriptor = model.preprocess(&DynamicImage::ImageRgb8(loco03));
        assert_eq!(descriptor.len(), 1568);
    }
}
