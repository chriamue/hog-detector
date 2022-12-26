#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// This module provides a struct and functions for working with bounding boxes.
pub mod bbox;
/// module for classifiers
pub mod classifier;
/// This module provides a trait for data augmentation.
/// Data augmentation is a technique used to increase the size of a dataset by creating modified versions of existing data.
pub mod data_augmentation;
/// The `dataset` module provides functionality for loading train datasets.
/// It contains functions and structs that can be used to read and parse data from a variety of sources.
pub mod dataset;
/// detection struct module
pub mod detection;
/// detector functionality
pub mod detector;
/// Feature descriptors are used in computer vision and image processing to describe the features of an image.
pub mod feature_descriptor;
/// hogdetector struct module
pub mod hogdetector;
/// predict functionality
pub mod predictable;
/// some structs and functions usable for tests
pub mod tests;
/// train functions
pub mod trainable;
/// This module provides a collection of functions for manipulating images, such as resizing, cropping, and rotating.
pub mod utils;

/// web assembly module
//#[cfg(target_arch = "wasm32")]
#[cfg(not(tarpaulin_include))]
#[cfg(feature = "wasm")]
pub mod wasm;

pub use dataset::DataSet;
pub use detector::Detector;
pub use hogdetector::HogDetector;
pub use predictable::Predictable;
pub use trainable::Trainable;

/// object class type
pub type Class = u32;
/// annotation is a object bounding box in image and class type
pub type Annotation = (bbox::BBox, Class);

/// the hog_detector prelude
pub mod prelude {
    pub use crate::bbox::BBox;
    pub use crate::classifier::Classifier;
    pub use crate::dataset::DataSet;
    pub use crate::detection::Detection;
    pub use crate::detector::Detector;
    pub use crate::feature_descriptor::FeatureDescriptor;
    pub use crate::predictable::Predictable;
    pub use crate::trainable::Trainable;
}
