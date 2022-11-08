use image::{DynamicImage, RgbImage};
use imageproc::hog::{hog, HogOptions};
use serde::{Deserialize, Serialize};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::svm::svc::SVC;
use smartcore::svm::LinearKernel;

#[derive(Serialize, Deserialize, Debug)]
pub struct HogDetector {
    pub svc: Option<SVC<f32, DenseMatrix<f32>, LinearKernel>>,
}

impl PartialEq for HogDetector {
    fn eq(&self, other: &HogDetector) -> bool {
        self.svc.is_none() && other.svc.is_none()
            || self.svc.is_some() && other.svc.is_some() && self.svc.eq(&other.svc)
    }
}

impl Default for HogDetector {
    fn default() -> Self {
        HogDetector { svc: None }
    }
}

impl HogDetector {
    pub fn preprocess(&self, image: &RgbImage) -> Vec<f32> {
        let luma = DynamicImage::ImageRgb8(image.clone()).to_luma8();
        let options = HogOptions {
            orientations: 8,
            signed: true,
            cell_side: 4,
            block_side: 2,
            block_stride: 1,
        };
        hog(&luma, options).unwrap()
    }

    pub fn preprocess_matrix(&self, images: Vec<RgbImage>) -> DenseMatrix<f32> {
        let descriptors: Vec<Vec<f32>> =
            images.iter().map(|image| self.preprocess(image)).collect();
        let samples = descriptors.len();
        let features = descriptors.first().unwrap().len();
        let descriptors: Vec<f32> = descriptors.into_iter().flatten().collect();
        DenseMatrix::from_vec(samples, features, &descriptors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops::resize, imageops::FilterType, open};

    #[test]
    fn test_default() {
        let model = HogDetector::default();
        assert!(model.svc.is_none());
    }

    #[test]
    fn test_part_eq() {
        let model1 = HogDetector::default();
        let model2 = HogDetector::default();
        assert!(model1.svc.is_none());
        assert!(model1.eq(&model2));
        assert!(model1.eq(&model1));
    }

    #[test]
    fn test_hog() {
        let model = HogDetector::default();
        let loco03 = open("res/loco03.jpg").unwrap().to_rgb8();
        let loco03 = resize(&loco03, 32, 32, FilterType::Nearest);
        let descriptor = model.preprocess(&loco03);
        assert_eq!(descriptor.len(), 1568);
    }
}
