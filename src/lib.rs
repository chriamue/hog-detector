#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// module for classifiers
pub mod classifier;
/// This module provides a trait for data augmentation.
/// Data augmentation is a technique used to increase the size of a dataset by creating modified versions of existing data.
pub mod data_augmentation;
/// The `dataset` module provides functionality for loading train datasets.
/// It contains functions and structs that can be used to read and parse data from a variety of sources.
pub mod dataset;
/// detector functionality
pub mod detector;
/// Feature descriptors are used in computer vision and image processing to describe the features of an image.
pub mod feature_descriptor;
/// hogdetector struct module
pub mod hogdetector;

/// This module provides a collection of functions for manipulating images, such as resizing, cropping, and rotating.
pub mod utils;

/// web assembly module
//#[cfg(target_arch = "wasm32")]
//#[cfg(not(tarpaulin_include))]
//#[cfg(feature = "wasm")]
//pub mod wasm;
pub use dataset::DataSet;
pub use detector::Detector;
pub use hogdetector::HogDetector;

/// the hog_detector prelude
pub mod prelude {
    pub use crate::classifier::Classifier;
    pub use crate::dataset::DataSet;
    pub use crate::detector::Detector;
    pub use object_detector_rust::bbox::BBox;
    pub use object_detector_rust::prelude::Detection;
    pub use object_detector_rust::prelude::Predictable;
    pub use object_detector_rust::prelude::Trainable;
    pub use object_detector_rust::types::AnnotatedImage;
    pub use object_detector_rust::types::Annotation;
    pub use object_detector_rust::types::Class;
}
