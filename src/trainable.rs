use crate::{DataSet, HogDetector};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::svm::svc::{SVCParameters, SVC};

pub trait Trainable {
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<f32>);
    fn train_class(&mut self, dataset: &DataSet, class: f32);
}

impl Trainable for HogDetector {
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<f32>) {
        let svc = SVC::fit(&x_train, &y_train, SVCParameters::default().with_c(10.0)).unwrap();
        self.svc = Some(svc);
    }

    fn train_class(&mut self, dataset: &DataSet, class: f32) {
        let (x_train, y_train, _, _) = dataset.get();
        let x_train = self.preprocess_matrix(x_train);
        let y_train = y_train
            .iter()
            .map(|y| if *y == class { *y } else { 0.0 })
            .collect();
        self.train(x_train, y_train);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train() {
        let mut model = HogDetector::default();

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
