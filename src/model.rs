use crate::dataset::DataSet;
use image::{DynamicImage, RgbImage};
use imageproc::hog::{hog, HogOptions};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::LinearKernel;

pub struct Model {
    pub options: HogOptions,
    pub svc: Option<SVC<f32, DenseMatrix<f32>, LinearKernel>>,
}

impl Default for Model {
    fn default() -> Self {
        let options = HogOptions {
            orientations: 9,
            signed: false,
            cell_side: 8,
            block_side: 2,
            block_stride: 2,
        };
        Model { options, svc: None }
    }
}

impl Model {
    pub fn preprocess(&self, image: &RgbImage) -> Vec<f32> {
        let luma = DynamicImage::ImageRgb8(image.clone()).to_luma8();
        let descriptor = hog(&luma, self.options).unwrap();
        descriptor
    }

    pub fn preprocess_matrix(&self, images: Vec<RgbImage>) -> DenseMatrix<f32> {
        let descriptors: Vec<Vec<f32>> =
            images.iter().map(|image| self.preprocess(image)).collect();
        let samples = descriptors.len();
        let features = descriptors.first().unwrap().len();
        let descriptors: Vec<f32> = descriptors.into_iter().flatten().collect();
        DenseMatrix::from_vec(samples, features, &descriptors)
    }

    pub fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<f32>) {
        let svc = SVC::fit(&x_train, &y_train, SVCParameters::default().with_c(10.0)).unwrap();
        self.svc = Some(svc);
    }

    pub fn train_class(&mut self, dataset: &DataSet, class: f32) {
        let (x_train, y_train, _, _) = dataset.get();
        let x_train = self.preprocess_matrix(x_train);
        let y_train = y_train
            .iter()
            .map(|y| if *y == class { *y } else { -1.0 })
            .collect();
        self.train(x_train, y_train);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops::resize, imageops::FilterType, open};

    #[test]
    fn test_default() {
        let model = Model::default();
        assert_eq!(model.options.orientations, 9);
    }

    #[test]
    fn test_hog() {
        let model = Model::default();
        let loco03 = open("res/loco03.jpg").unwrap().to_rgb8();
        let loco03 = resize(&loco03, 32, 32, FilterType::Nearest);
        let descriptor = model.preprocess(&loco03);
        assert_eq!(descriptor.len(), 144);
    }

    #[test]
    fn test_train() {
        let mut model = Model::default();

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(false);

        model.train_class(&dataset, 5.0);
        assert!(model.svc.is_some());
    }
}
