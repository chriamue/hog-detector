use image::{DynamicImage, RgbImage};
use imageproc::hog::{hog, HogOptions};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::svm::svc::SVC;
use smartcore::svm::LinearKernel;

pub struct HogDetector {
    pub options: HogOptions,
    pub svc: Option<SVC<f32, DenseMatrix<f32>, LinearKernel>>,
}

impl Default for HogDetector {
    fn default() -> Self {
        let options = HogOptions {
            orientations: 9,
            signed: false,
            cell_side: 8,
            block_side: 2,
            block_stride: 2,
        };
        HogDetector { options, svc: None }
    }
}

impl HogDetector {
    pub fn preprocess(&self, image: &RgbImage) -> Vec<f32> {
        let luma = DynamicImage::ImageRgb8(image.clone()).to_luma8();
        hog(&luma, self.options).unwrap()
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
        assert_eq!(model.options.orientations, 9);
    }

    #[test]
    fn test_hog() {
        let model = HogDetector::default();
        let loco03 = open("res/loco03.jpg").unwrap().to_rgb8();
        let loco03 = resize(&loco03, 32, 32, FilterType::Nearest);
        let descriptor = model.preprocess(&loco03);
        assert_eq!(descriptor.len(), 144);
    }
}
