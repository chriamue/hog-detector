/// naive bayes classifier module
pub mod bayes;

#[cfg(feature = "svm")]
/// svm classifier module
pub mod svm;
pub use bayes::BayesClassifier;
pub use object_detector_rust::prelude::Classifier;
