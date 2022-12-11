use crate::{
    classifier::Classifier,
    feature_descriptor::{FeatureDescriptor, HogFeatureDescriptor},
    Detector, Trainable,
};
use image::DynamicImage;
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
pub struct HogDetector<C: Classifier> {
    /// support vector classifier
    pub classifier: Option<C>,
    /// the feature descriptor
    pub feature_descriptor: Box<dyn FeatureDescriptor>,
}

/// trait of an hog detector
pub trait HogDetectorTrait: Trainable + Detector + Send + Sync {
    /// save to string
    fn save(&self) -> String;
    /// load from string
    fn load(&mut self, model: &str);
    /// reference to detector trait
    fn detector(&self) -> &dyn Detector;
}

unsafe impl<C: Classifier> Send for HogDetector<C> {}
unsafe impl<C: Classifier> Sync for HogDetector<C> {}

impl<C: Classifier> PartialEq for HogDetector<C> {
    fn eq(&self, other: &HogDetector<C>) -> bool {
        self.classifier.is_none() && other.classifier.is_none()
            || self.classifier.is_some()
                && other.classifier.is_some()
                && self.classifier.eq(&other.classifier)
    }
}

impl<C: Classifier> Default for HogDetector<C> {
    fn default() -> Self {
        HogDetector::<C> {
            classifier: None,
            feature_descriptor: Box::new(HogFeatureDescriptor::default()),
        }
    }
}

impl<C: Classifier> HogDetector<C> {
    /// preprocesses image to vector
    pub fn preprocess(&self, image: &DynamicImage) -> Vec<f32> {
        self.feature_descriptor.calculate_feature(image).unwrap()
    }
    /// preprocesses images into dense matrix
    pub fn preprocess_matrix(&self, images: Vec<DynamicImage>) -> DenseMatrix<f32> {
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
        let descriptor = model.preprocess(&DynamicImage::ImageRgb8(loco03));
        assert_eq!(descriptor.len(), 1568);
    }
}
