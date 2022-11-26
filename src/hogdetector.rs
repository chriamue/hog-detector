use crate::{
    classifier::Classifier,
    feature_descriptor::{FeatureDescriptor, HogFeatureDescriptor},
    Detector, Trainable,
};
use image::{DynamicImage, RgbImage};
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Hog Detector struct
#[derive(Serialize, Deserialize, Debug)]
pub struct HogDetector<C: Classifier> {
    /// support vector classifier
    pub classifier: Option<C>,
}

/// trait of an hog detector
pub trait HogDetectorTrait: Trainable + Detector {}

impl<C: Classifier> PartialEq for HogDetector<C> {
    fn eq(&self, other: &HogDetector<C>) -> bool {
        self.classifier.is_none() && other.classifier.is_none()
            || self.classifier.is_some()
                && other.classifier.is_some()
                && self.classifier.eq(&other.classifier)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl<C: Classifier> Default for HogDetector<C> {
    fn default() -> Self {
        HogDetector::<C> { classifier: None }
    }
}

impl<C: Classifier> HogDetector<C> {
    /// preprocesses image to vector
    pub fn preprocess(&self, image: &RgbImage) -> Vec<f32> {
        let luma = DynamicImage::ImageRgb8(image.clone()).to_luma8();
        let descriptor = HogFeatureDescriptor::default();
        descriptor.calculate_feature(luma).unwrap()
    }
    /// preprocesses images into dense matrix
    pub fn preprocess_matrix(&self, images: Vec<RgbImage>) -> DenseMatrix<f32> {
        let descriptors: Vec<Vec<f32>> =
            images.iter().map(|image| self.preprocess(image)).collect();
        DenseMatrix::from_2d_vec(&descriptors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::SVMClassifier;
    use image::{imageops::resize, imageops::FilterType, open};

    #[test]
    fn test_default() {
        let model = HogDetector::<SVMClassifier>::default();
        assert!(model.classifier.is_none());
    }

    #[test]
    fn test_part_eq() {
        let model1 = HogDetector::<SVMClassifier>::default();
        let model2 = HogDetector::<SVMClassifier>::default();
        assert!(model1.classifier.is_none());
        assert!(model1.eq(&model2));
        assert!(model1.eq(&model1));
    }

    #[test]
    fn test_hog() {
        let model = HogDetector::<SVMClassifier>::default();
        let loco03 = open("res/loco03.jpg").unwrap().to_rgb8();
        let loco03 = resize(&loco03, 32, 32, FilterType::Nearest);
        let descriptor = model.preprocess(&loco03);
        assert_eq!(descriptor.len(), 1568);
    }
}
