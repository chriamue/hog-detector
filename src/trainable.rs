use crate::classifier::svm::{SVCParametersType, SVCType};
use crate::classifier::SVMClassifier;
use crate::{DataSet, HogDetector, Predictable};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::svc::{SVCParameters, SVC};

/// trainable trait
pub trait Trainable {
    /// trains on given data
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<u32>);
    /// train class on given dataset
    fn train_class(&mut self, dataset: &dyn DataSet, class: u32);
    /// evaluate class on dataset
    fn evaluate(&mut self, dataset: &dyn DataSet, class: u32) -> f32;
}

impl<'a> Trainable for HogDetector<SVMClassifier<'a>> {
    fn train(&mut self, x_train: DenseMatrix<f32>, y_train: Vec<u32>) {
        let parameters: SVCParametersType = SVCParameters::default();

        let svc = SVC::fit(&x_train, &y_train, &parameters).unwrap();
        let deserialized_svc: SVCType<'a> =
            serde_json::from_str(&serde_json::to_string(&svc).unwrap()).unwrap();
        let classifier = SVMClassifier {
            svc: Some(deserialized_svc),
        };
        self.classifier = Some(classifier);
    }

    fn train_class(&mut self, dataset: &dyn DataSet, class: u32) {
        let (x_train, y_train, _, _) = dataset.get();
        let x_train = self.preprocess_matrix(x_train);
        let y_train = y_train
            .iter()
            .map(|y| if *y as u32 == class { *y } else { 0u32 })
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

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{classifier::SVMClassifier, dataset::MemoryDataSet};

    #[test]
    fn test_train() {
        let mut model = HogDetector::<SVMClassifier>::default();

        let mut dataset = MemoryDataSet::new_test();
        dataset.load();

        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());
    }

    #[test]
    fn test_evaluate() {
        let mut model = HogDetector::<SVMClassifier>::default();

        let mut dataset = MemoryDataSet::new_test();
        dataset.load();

        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());
        assert!(model.evaluate(&dataset, 1) > 0.0);
    }
}
