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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::window_crop;
    use crate::folder_dataset::FolderDataSet;
    use crate::trainable::Trainable;
    use crate::DataSet;

    #[test]
    fn test_predict() {
        let mut model = HogDetector::default();

        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(false);
        dataset.generate_random_annotations(5);

        model.train_class(&dataset, 5);
        assert!(model.svc.is_some());
        let loco03 = image::open("res/loco03.jpg").unwrap().to_rgb8();
        let loco03 = window_crop(&loco03, 32, 32, (60, 35));

        let predicted = model.predict(&loco03);
        assert_eq!(predicted, 5);
    }
}
