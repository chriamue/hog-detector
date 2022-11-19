use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::svc::{SVCParameters, SVC};

use super::Classifier;

/// svc type for float x and unsigned integer y
pub type SVCType<'a> = SVC<'a, f32, u32, DenseMatrix<f32>, Vec<u32>>;
/// svc parameters type for float x and unsigned integer y
pub type SVCParametersType = SVCParameters<f32, u32, DenseMatrix<f32>, Vec<u32>>;

/// A Support Vector Machine classifier
#[derive(Serialize, Deserialize, Debug)]
pub struct SVMClassifier<'a> {
    /// svc classifier
    pub svc: Option<SVCType<'a>>,
}

impl Classifier for SVMClassifier<'_> {}

impl PartialEq for SVMClassifier<'_> {
    fn eq(&self, other: &SVMClassifier) -> bool {
        self.svc.is_none() && other.svc.is_none()
            || self.svc.is_some() && other.svc.is_some() && self.svc.eq(&other.svc)
    }
}

impl Default for SVMClassifier<'_> {
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
