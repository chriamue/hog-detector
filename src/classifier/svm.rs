use serde::{Deserialize, Serialize};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::svm::svc::SVC;
use smartcore::svm::LinearKernel;

use super::Classifier;

/// A Support Vector Machine classifier
#[derive(Serialize, Deserialize, Debug)]
pub struct SVMClassifier {
    /// svc classifier
    pub svc: Option<SVC<f32, DenseMatrix<f32>, LinearKernel>>,
}

impl Classifier for SVMClassifier {}

impl PartialEq for SVMClassifier {
    fn eq(&self, other: &SVMClassifier) -> bool {
        self.svc.is_none() && other.svc.is_none()
            || self.svc.is_some() && other.svc.is_some() && self.svc.eq(&other.svc)
    }
}

impl Default for SVMClassifier {
    fn default() -> Self {
        SVMClassifier { svc: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let classifier = SVMClassifier::default();
        assert!(classifier.svc.is_none());
    }

    #[test]
    fn test_part_eq() {
        let classifier1 = SVMClassifier::default();
        let classifier2 = SVMClassifier::default();
        assert!(classifier1.svc.is_none());
        assert!(classifier1.eq(&classifier2));
        assert!(classifier1.eq(&classifier1));
    }
}
