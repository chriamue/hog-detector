use crate::HogDetector;
use image::{imageops::resize, imageops::FilterType};
use image::{DynamicImage, RgbImage};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

pub trait Predictable {
    fn predict(&self, image: &RgbImage) -> Vec<f32>;
}

impl Predictable for HogDetector {
    fn predict(&self, image: &RgbImage) -> Vec<f32> {
        let image = resize(
            &DynamicImage::ImageRgb8(image.clone()),
            32,
            32,
            FilterType::Nearest,
        );
        let image = DynamicImage::ImageRgba8(image).to_rgb8();
        let x = self.preprocess(&image);
        let x = DenseMatrix::from_vec(1, x.len(), &x);
        let y = self.svc.as_ref().unwrap().predict(&x).unwrap();
        y
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainable::Trainable;
    use crate::DataSet;

    #[test]
    fn test_predict() {
        let mut model = HogDetector::default();

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(false);

        model.train_class(&dataset, 5.0);
        assert!(model.svc.is_some());
        let loco03 = image::open("res/loco03.jpg").unwrap().to_rgb8();

        let predicted = model.predict(&loco03);
        assert_eq!(predicted, vec![5.0]);
    }
}
