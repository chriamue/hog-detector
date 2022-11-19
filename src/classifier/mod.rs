mod svm;
pub use svm::SVMClassifier;
mod randomforest;
pub use randomforest::RandomForestClassifier;

/// A classifier trait
pub trait Classifier: PartialEq {}
