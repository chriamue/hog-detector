use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::DataSet;

/// trainable trait
pub trait Trainable {
    /// trains on given data
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<u32>);
    /// train class on given dataset
    fn train_class(&mut self, dataset: &dyn DataSet, class: u32);
    /// evaluate class on dataset
    fn evaluate(&mut self, dataset: &dyn DataSet, class: u32) -> f32;
}
