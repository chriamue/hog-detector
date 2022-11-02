use crate::{DataSet, HogDetector, Predictable};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::svm::svc::{SVCParameters, SVC};

pub trait Trainable {
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<f32>);
    fn train_class(&mut self, dataset: &dyn DataSet, class: u32);
    fn evaluate(&mut self, dataset: &dyn DataSet, class: u32) -> f32;
}

impl Trainable for HogDetector {
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<f32>) {
        let svc = SVC::fit(
            &x_train,
            &y_train,
            SVCParameters::default().with_c(10.0).with_epoch(3),
        )
        .unwrap();
        self.svc = Some(svc);
    }

    fn train_class(&mut self, dataset: &dyn DataSet, class: u32) {
        let (x_train, y_train, _, _) = dataset.get();
        let x_train = self.preprocess_matrix(x_train);
        let y_train = y_train
            .iter()
            .map(|y| if *y as u32 == class { *y as f32 } else { 0.0 })
            .collect();
        self.train(x_train, y_train);
    }

    fn evaluate(&mut self, dataset: &dyn DataSet, class: u32) -> f32 {
        let mut i = 0;
        let (x_train, y_train, _, _) = dataset.get();
        x_train.iter().zip(y_train).for_each(|(img, y)| {
            let pred = self.predict(img);
            if (pred == y && y == class) || (pred == 0 && y != class) {
                i += 1;
            }
        });
        i as f32 / x_train.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::FolderDataSet;

    #[test]
    fn test_train() {
        let mut model = HogDetector::default();

        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(false);

        model.train_class(&dataset, 5);
        assert!(model.svc.is_some());
    }

    #[test]
    fn test_evaluate() {
        let mut model = HogDetector::default();

        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(true);

        model.train_class(&dataset, 5);
        assert!(model.svc.is_some());
        println!("{:?}", model.evaluate(&dataset, 5));
        assert!(model.evaluate(&dataset, 5) > 0.0);
    }
}
