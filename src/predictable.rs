use crate::classifier::SVMClassifier;
use crate::HogDetector;
use image::{imageops::resize, imageops::FilterType};
use image::{DynamicImage, RgbImage};
use smartcore::linalg::basic::matrix::DenseMatrix;
/// predictable trait
pub trait Predictable {
    /// predicts class of image
    fn predict(&self, image: &RgbImage) -> u32;
}

impl<'a> Predictable for HogDetector<SVMClassifier<'a>> {
    fn predict(&self, image: &RgbImage) -> u32 {
        let image = resize(
            &DynamicImage::ImageRgb8(image.clone()),
            32,
            32,
            FilterType::Gaussian,
        );
        let image = DynamicImage::ImageRgba8(image).to_rgb8();
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

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {}
