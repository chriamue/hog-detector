mod svm;
pub use svm::SVMClassifier;

/// A classifier trait
pub trait Classifier: PartialEq {}
