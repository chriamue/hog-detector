use crate::HogDetector;
use image::{imageops::resize, imageops::FilterType};
use image::{DynamicImage, RgbImage};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

pub trait Predictable {
    fn predict(&self, image: &RgbImage) -> u32;
}

impl Predictable for HogDetector {
    fn predict(&self, image: &RgbImage) -> u32 {
        let image = resize(
            &DynamicImage::ImageRgb8(image.clone()),
            32,
            32,
            FilterType::Gaussian,
        );
        let image = DynamicImage::ImageRgba8(image).to_rgb8();
        let x = self.preprocess(&image);
        let x = DenseMatrix::from_vec(1, x.len(), &x);
        let y = self.svc.as_ref().unwrap().predict(&x).unwrap();
        *y.first().unwrap() as u32
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {}
