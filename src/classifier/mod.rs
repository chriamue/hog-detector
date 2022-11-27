/// naive bayes classifier module
pub mod bayes;
pub use bayes::BayesClassifier;
/// combined classifier of two classifiers
pub mod combined;
pub use combined::CombinedClassifier;
/// svm classifier module
pub mod svm;
pub use svm::SVMClassifier;
/// random forest classifier module
pub mod randomforest;
pub use randomforest::RandomForestClassifier;

/// A classifier trait
pub trait Classifier: PartialEq {}
